package main

// The Go side of the real-file quality check: `oracle -quality <ppm-dir> <out-dir>` (docs/quality.md).
// Per binary P6 tile <stem>.ppm it writes the outputs of seven raster operations; examples/quality.rs
// does the same with the Rust crate and must write the same bytes.

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	raster "github.com/dobbo-ca/raster-go"
)

const maxTilePixels = 1 << 26 // a larger tile is an error: the oracle bounds its own allocations

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f'
}

// parseP6 reads a binary PPM with maxval 255: `P6 <w> <h> 255`, one whitespace byte, then
// exactly w*h*3 bytes.
func parseP6(b []byte) (w, h int, pix []byte, err error) {
	pos := 0
	tok := func() string {
		for pos < len(b) && isSpace(b[pos]) {
			pos++
		}
		s := pos
		for pos < len(b) && !isSpace(b[pos]) {
			pos++
		}
		return string(b[s:pos])
	}
	magic, ws, hs, maxval := tok(), tok(), tok(), tok()
	if magic != "P6" || maxval != "255" || pos >= len(b) {
		return 0, 0, nil, errors.New("not a P6 PPM with maxval 255")
	}
	pos++
	w, werr := strconv.Atoi(ws)
	h, herr := strconv.Atoi(hs)
	if werr != nil || herr != nil || w <= 0 || h <= 0 || w > maxTilePixels || h > maxTilePixels || w*h > maxTilePixels {
		return 0, 0, nil, errors.New("bad PPM size")
	}
	if len(b)-pos != 3*w*h {
		return 0, 0, nil, errors.New("PPM data length does not match its size")
	}
	return w, h, b[pos:], nil
}

func newPlane(w, h, ch int) raster.Plane {
	return raster.Plane{Pix: make([]byte, w*h*ch), W: w, H: h, Stride: w * ch, Channels: ch}
}

func qualityDir(in, out string) error {
	ents, err := os.ReadDir(in)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	lim := raster.Limits{MaxPixels: maxTilePixels, MaxScratchBytes: 1 << 30, MaxOutputBytes: 1 << 28}
	n := 0
	for _, e := range ents {
		if !e.Type().IsRegular() || !strings.HasSuffix(e.Name(), ".ppm") {
			continue
		}
		if err := qualityTile(filepath.Join(in, e.Name()), out, strings.TrimSuffix(e.Name(), ".ppm"), lim); err != nil {
			return fmt.Errorf("%s: %w", e.Name(), err)
		}
		n++
	}
	fmt.Println(n, "tiles")
	return nil
}

func qualityTile(path, out, stem string, lim raster.Limits) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	w, h, pix, err := parseP6(b)
	if err != nil {
		return err
	}
	ctx := context.Background()
	rgb := raster.Plane{Pix: pix, W: w, H: h, Stride: 3 * w, Channels: 3}
	grey := newPlane(w, h, 1)
	if err := raster.Luma(ctx, grey, rgb, lim); err != nil {
		return err
	}
	put := func(ext string, p raster.Plane) error {
		magic := "P6"
		if p.Channels == 1 {
			magic = "P5"
		}
		hdr := fmt.Sprintf("%s\n%d %d\n255\n", magic, p.W, p.H)
		return os.WriteFile(filepath.Join(out, stem+"."+ext), append([]byte(hdr), p.Pix...), 0o644)
	}
	half := func(ext string, src raster.Plane) error {
		if src.W < 2 || src.H < 2 {
			return nil
		}
		d := newPlane(src.W/2, src.H/2, src.Channels)
		if err := raster.Scale2to1(ctx, d, src, lim); err != nil {
			return err
		}
		return put(ext, d)
	}
	resize := func(ext string, src raster.Plane, dw, dh int) error {
		d := newPlane(dw, dh, src.Channels)
		if err := raster.Resample(ctx, d, src, lim); err != nil {
			return err
		}
		return put(ext, d)
	}
	for _, step := range []func() error{
		func() error { return half("s2rgb.ppm", rgb) },
		func() error { return put("luma.pgm", grey) },
		func() error { return half("s2gray.pgm", grey) },
		func() error { return resize("rsrgb.ppm", rgb, max(1, w*3/7), max(1, h*3/7)) },
		func() error {
			if (w*3/2)*(h*3/2) > 1<<22 {
				return nil
			}
			return resize("rsup.pgm", grey, w*3/2, h*3/2)
		},
		func() error {
			d := newPlane(h, w, 3)
			if err := raster.Rotate90(ctx, d, rgb, true, lim); err != nil {
				return err
			}
			return put("rot.ppm", d)
		},
		func() error {
			d := newPlane(h, w, 1)
			if err := raster.Rotate90(ctx, d, grey, true, lim); err != nil {
				return err
			}
			return put("rotg.pgm", d)
		},
	} {
		if err := step(); err != nil {
			return err
		}
	}
	return nil
}
