// Command oracle is the Go side of the parity check (see docs/parity.md).
//
//	oracle -gen <inputs>              write the test inputs
//	oracle <inputs> <outputs>         run every input
//	oracle -bench N                   time the kernels on a 2550x3300 page
//	oracle -quality <ppms> <outputs>  the real-file quality check (quality.go)
//
// Run mode writes, per input, <name>.out (the destination bytes) or <name>.err (a short error
// class) and <name>.partial.out (the destination bytes left behind by the failed call).
// examples/oracle.rs does the same with the Rust crate. The input format is in docs/parity.md.
package main

import (
	"context"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	raster "github.com/dobbo-ca/raster-go"
)

const (
	hdrLen = 33      // op lim flags sch dch, then sw sh sst dw dh dst dlen as u32 LE
	maxDst = 1 << 21 // a larger dlen is bad-input: the oracle bounds its own allocations
)

type tc struct {
	op, lim, flags, sch, dch       byte
	sw, sh, sst, dw, dh, dst, dlen uint32
	pix                            []byte
}

func (c tc) bytes() []byte {
	b := []byte{c.op, c.lim, c.flags, c.sch, c.dch}
	for _, v := range [...]uint32{c.sw, c.sh, c.sst, c.dw, c.dh, c.dst, c.dlen} {
		b = binary.LittleEndian.AppendUint32(b, v)
	}
	return append(b, c.pix...)
}

func parse(b []byte) (tc, bool) {
	if len(b) < hdrLen || b[0] > 4 {
		return tc{}, false
	}
	u := func(i int) uint32 { return binary.LittleEndian.Uint32(b[5+4*i:]) }
	c := tc{op: b[0], lim: b[1], flags: b[2], sch: b[3], dch: b[4],
		sw: u(0), sh: u(1), sst: u(2), dw: u(3), dh: u(4), dst: u(5), dlen: u(6), pix: b[hdrLen:]}
	return c, c.dlen <= maxDst
}

// limits maps the lim byte; scratch is capped at 64 MiB so the oracle stays small.
func limits(sel byte) raster.Limits {
	l := raster.Limits{MaxPixels: 1 << 26, MaxScratchBytes: 64 << 20, MaxOutputBytes: 1 << 28}
	switch sel {
	case 1:
		l.MaxPixels = 4095
	case 2:
		l.MaxScratchBytes = 1 << 20
	case 3:
		l.MaxOutputBytes = 1023
	case 4:
		l.MaxPixels, l.MaxOutputBytes = 4096, 1024
	}
	return l
}

func class(err error) string {
	var le *raster.LimitError
	switch {
	case errors.Is(err, context.Canceled):
		return "cancelled"
	case errors.Is(err, raster.ErrInvalid):
		return "invalid"
	case errors.As(err, &le):
		switch le.Name {
		case "pixels":
			return "limit-pixels"
		case "scratch bytes":
			return "limit-scratch"
		case "output bytes":
			return "limit-output"
		}
	}
	return "other:" + err.Error()
}

// failAt is a context whose Nth Err call reports Canceled: a cancel part-way through a call.
type failAt struct {
	context.Context
	n int
}

func (f *failAt) Err() error {
	if f.n--; f.n == 0 {
		return context.Canceled
	}
	return f.Context.Err()
}

func run(c tc) ([]byte, error) {
	src := raster.Plane{Pix: c.pix, W: int(c.sw), H: int(c.sh), Stride: int(c.sst), Channels: int(c.sch)}
	dstPix := make([]byte, c.dlen)
	dst := raster.Plane{Pix: dstPix, W: int(c.dw), H: int(c.dh), Stride: int(c.dst), Channels: int(c.dch)}
	bg, cancel := context.WithCancel(context.Background())
	defer cancel()
	if c.flags&1 != 0 {
		cancel()
	}
	var ctx context.Context = bg
	if n := int(c.flags >> 1); n > 0 {
		ctx = &failAt{bg, n}
	}
	lim := limits(c.lim)
	var err error
	switch c.op {
	case 0:
		err = raster.Scale2to1(ctx, dst, src, lim)
	case 1:
		err = raster.Luma(ctx, dst, src, lim)
	case 2, 3:
		err = raster.Rotate90(ctx, dst, src, c.op == 2, lim)
	case 4:
		err = raster.Resample(ctx, dst, src, lim)
	}
	return dstPix, err
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "oracle:", err)
		os.Exit(1)
	}
}

func main() {
	gen := flag.Bool("gen", false, "write the inputs: -gen <inputs>")
	bench := flag.Int("bench", 0, "time the kernels N times: -bench N")
	quality := flag.Bool("quality", false, "run the quality gate: -quality <ppm-dir> <outputs>")
	flag.Parse()
	a := flag.Args()
	switch {
	case *quality && !*gen && len(a) == 2:
		check(qualityDir(a[0], a[1]))
	case *bench > 0 && len(a) == 0:
		benchAll(*bench)
	case *gen && len(a) == 1:
		check(generate(a[0]))
	case !*gen && !*quality && len(a) == 2:
		check(runDir(a[0], a[1]))
	default:
		fmt.Fprintln(os.Stderr, "usage: oracle -gen <inputs> | oracle <inputs> <outputs> | oracle -bench N | oracle -quality <ppms> <outputs>")
		os.Exit(2)
	}
}

func runDir(in, out string) error {
	ents, err := os.ReadDir(in)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}
	for _, e := range ents {
		b, err := os.ReadFile(filepath.Join(in, e.Name()))
		if err != nil {
			return err
		}
		base := filepath.Join(out, e.Name())
		c, ok := parse(b)
		if !ok {
			if err := os.WriteFile(base+".err", []byte("bad-input\n"), 0o644); err != nil {
				return err
			}
			continue
		}
		dst, rerr := run(c)
		if rerr == nil {
			err = os.WriteFile(base+".out", dst, 0o644)
		} else if err = os.WriteFile(base+".err", []byte(class(rerr)+"\n"), 0o644); err == nil {
			err = os.WriteFile(base+".partial.out", dst, 0o644)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// ---- generators ----------------------------------------------------------------------------

// bytesOf is raster-go's fixture.Bytes (xorshift64*).
func bytesOf(seed uint64, n int) []byte {
	s := seed*0x9E3779B97F4A7C15 + 1
	b := make([]byte, n)
	for i := range b {
		s ^= s >> 12
		s ^= s << 25
		s ^= s >> 27
		b[i] = byte((s * 0x2545F4914F6CDD1D) >> 56)
	}
	return b
}

// page is raster-go's fixture.Page.
func page(seed uint64, w, h, ch int) []byte {
	b := bytesOf(seed, w*h*ch)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for c := 0; c < ch; c++ {
				i := (y*w+x)*ch + c
				switch (x/16 + y/16) % 5 {
				case 0:
					b[i] = 255
				case 1:
					b[i] = byte(x*3 + y*5 + c*40)
				case 2:
					b[i] = 0
				default:
					b[i] = b[i]&0x1f | byte(x+y)
				}
			}
		}
	}
	return b
}

func size(w, h, ch, pad int) int { return (h-1)*(w*ch+pad) + w*ch }

// padded is the test's plane(): the page in a buffer with pad spare bytes (0xEE) per row.
func padded(seed uint64, w, h, ch, pad int) []byte {
	stride := w*ch + pad
	buf := make([]byte, size(w, h, ch, pad))
	for i := range buf {
		buf[i] = 0xEE
	}
	fill := page(seed, w, h, ch)
	for y := 0; y < h; y++ {
		copy(buf[y*stride:], fill[y*w*ch:(y+1)*w*ch])
	}
	return buf
}

// mk is one call: src is the seed page at sw x sh x sch with spad, dst a blank dw x dh x dch with dpad.
func mk(op byte, sw, sh, sch, spad, dw, dh, dch, dpad int, seed uint64) tc {
	return tc{op: op, sch: byte(sch), dch: byte(dch),
		sw: uint32(sw), sh: uint32(sh), sst: uint32(sw*sch + spad),
		dw: uint32(dw), dh: uint32(dh), dst: uint32(dw*dch + dpad), dlen: uint32(size(dw, dh, dch, dpad)),
		pix: padded(seed, sw, sh, sch, spad)}
}

func flat(op byte, w, h, ch int, v byte, dw, dh int) tc {
	c := mk(op, w, h, ch, 0, dw, dh, ch, 0, 0)
	for i := range c.pix {
		c.pix[i] = v
	}
	return c
}

type rng uint64

func (r *rng) next() int {
	*r ^= *r << 13
	*r ^= *r >> 7
	*r ^= *r << 17
	return int(*r >> 11)
}

var widths = []int{1, 2, 3, 15, 16, 17, 18, 19, 31, 32, 33, 34, 35, 63, 64, 65, 66, 67, 127, 128, 129, 130, 255, 257}

func generate(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	var werr error
	count := 0
	addRaw := func(name string, b []byte) {
		if werr == nil {
			werr = os.WriteFile(filepath.Join(dir, name+".in"), b, 0o644)
		}
		count++
	}
	add := func(name string, c tc) { addRaw(name, c.bytes()) }

	// TestScale2to1, TestLuma, TestLumaExtremes, TestRotate90, TestResample.
	for _, ch := range []int{1, 3} {
		for _, w := range widths {
			for _, pad := range []int{0, 3} {
				add(fmt.Sprintf("scale-c%d-w%d-p%d", ch, w, pad), mk(0, w, 5, ch, pad, w/2, 2, ch, pad, uint64(w)))
			}
		}
	}
	for _, w := range widths {
		for _, pad := range []int{0, 3} {
			add(fmt.Sprintf("luma-w%d-p%d", w, pad), mk(1, w, 3, 3, pad, w, 3, 1, pad, uint64(w)))
		}
	}
	for _, v := range []byte{0, 255} {
		c := mk(1, 40, 1, 3, 0, 40, 1, 1, 0, 0)
		for i := range c.pix {
			c.pix[i] = v
		}
		add(fmt.Sprintf("luma-extreme-%d", v), c)
	}
	for _, ch := range []int{1, 3} {
		for _, wh := range [][2]int{{1, 1}, {1, 7}, {7, 1}, {16, 16}, {17, 33}, {40, 5}} {
			for _, cw := range []bool{true, false} {
				op := byte(3)
				if cw {
					op = 2
				}
				add(fmt.Sprintf("rotate-c%d-%dx%d-cw%v", ch, wh[0], wh[1], cw), mk(op, wh[0], wh[1], ch, 3, wh[1], wh[0], ch, 1, 9))
			}
		}
	}
	for _, ch := range []int{1, 3} {
		add(fmt.Sprintf("resample-identity-c%d", ch), mk(4, 37, 29, ch, 2, 37, 29, ch, 0, 5))
		for _, v := range []byte{0, 77, 255} {
			for _, d := range [][2]int{{32, 24}, {5, 3}, {1, 1}, {100, 90}, {64, 1}} {
				add(fmt.Sprintf("resample-flat%d-c%d-%dx%d", v, ch, d[0], d[1]), flat(4, 64, 48, ch, v, d[0], d[1]))
			}
		}
	}
	ramp := mk(4, 256, 4, 1, 0, 128, 2, 1, 0, 0)
	for y := 0; y < 4; y++ {
		for x := 0; x < 256; x++ {
			ramp.pix[y*256+x] = byte(x)
		}
	}
	add("resample-ramp", ramp)

	// TestInvalid: 7 bad planes as the src and dst of scale2to1 and the dst of resample.
	type bad struct{ n, w, h, st, ch uint32 }
	bads := []bad{{0, 0, 0, 0, 0}, {64, 8, 8, 7, 1}, {63, 8, 8, 8, 1}, {64, 8, 8, 8, 2},
		{64, 1<<32 - 1, 2, 1<<32 - 1, 1}, {64, 8, 1<<32 - 1, 8, 1}, {64, 1, 2, 1<<32 - 1, 1}}
	ok := func(op byte) tc { return mk(op, 8, 8, 1, 0, 4, 4, 1, 0, 0) }
	for i, p := range bads {
		c := ok(0) // bad src
		c.sw, c.sh, c.sst, c.sch, c.pix = p.w, p.h, p.st, byte(p.ch), make([]byte, p.n)
		add(fmt.Sprintf("invalid%d-src", i), c)
		c = ok(0) // bad dst
		c.dw, c.dh, c.dst, c.dch, c.dlen = p.w, p.h, p.st, byte(p.ch), p.n
		add(fmt.Sprintf("invalid%d-dst", i), c)
		c = ok(4) // bad dst for resample
		c.dw, c.dh, c.dst, c.dch, c.dlen = p.w, p.h, p.st, byte(p.ch), p.n
		add(fmt.Sprintf("invalid%d-resample", i), c)
	}
	add("invalid-dst-size", mk(0, 8, 8, 1, 0, 8, 8, 1, 0, 0))

	// TestLimits and TestResampleScratchLimit (lim 1..4 are in limits()).
	for _, l := range []byte{1, 3, 4} {
		c := mk(0, 64, 64, 1, 0, 32, 32, 1, 0, 1)
		c.lim = l
		add(fmt.Sprintf("limit-scale-%d", l), c)
	}
	big := tc{op: 1, sch: 3, dch: 1, sw: 1 << 14, sh: 1<<12 + 1, sst: 3 << 14, dw: 1 << 14, dh: 1<<12 + 1, dst: 1 << 14, pix: make([]byte, 1)}
	add("limit-short-buffer", big)
	c := mk(4, 2000, 2000, 1, 0, 1000, 1000, 1, 0, 1)
	c.lim = 2
	add("limit-resample-scratch", c)
	// TestResampleScratchExact: the two thin shapes (the 3000x2000 one is in the Rust test only).
	add("resample-thin-a", mk(4, 200000, 4, 1, 0, 1, 1, 1, 0, 1))
	add("resample-thin-b", mk(4, 4, 200000, 1, 0, 1, 1, 1, 0, 1))

	// TestCancel (the context is cancelled before the call).
	for i, c := range []tc{
		mk(0, 64, 64, 1, 0, 32, 32, 1, 0, 1), mk(1, 64, 64, 3, 0, 64, 64, 1, 0, 1),
		mk(2, 64, 64, 1, 0, 64, 64, 1, 0, 1), mk(4, 64, 64, 1, 0, 32, 32, 1, 0, 1),
	} {
		c.flags = 1
		add(fmt.Sprintf("cancel-%d", i), c)
	}

	// A cancel part-way through (flags>>1 = the Nth ctx.Err call fails): the rows done stay in dst.
	// Go checks once per dst row (scale, luma), per 16-row tile band (rotate), per src row and
	// then per dst row (resample), so 64x64 gives 32, 64, 4 and 64+32 checks.
	for _, m := range []struct {
		name string
		c    tc
		ns   []byte
	}{
		{"scale", mk(0, 64, 64, 1, 0, 32, 32, 1, 0, 1), []byte{1, 2, 17, 32, 33}},
		{"scale-rgb", mk(0, 64, 64, 3, 0, 32, 32, 3, 0, 1), []byte{5, 32}},
		{"luma", mk(1, 64, 64, 3, 0, 64, 64, 1, 0, 1), []byte{2, 40, 64, 65}},
		{"rotate", mk(2, 64, 64, 1, 0, 64, 64, 1, 0, 1), []byte{1, 2, 4, 5}},
		{"rotate-ccw", mk(3, 64, 64, 3, 0, 64, 64, 3, 0, 1), []byte{3}},
		{"resample", mk(4, 64, 64, 1, 0, 32, 32, 1, 0, 1), []byte{1, 30, 64, 65, 70, 96, 97}},
		{"resample-rgb", mk(4, 64, 64, 3, 0, 32, 32, 3, 0, 1), []byte{65, 80}},
	} {
		for _, n := range m.ns {
			c := m.c
			c.flags = n << 1
			add(fmt.Sprintf("cancel-mid-%s-%d", m.name, n), c)
		}
	}

	// TestGolden: the 30 golden cases.
	for _, sz := range [][2]int{{257, 131}, {640, 480}} {
		w, h := sz[0], sz[1]
		for _, ch := range []int{1, 3} {
			add(fmt.Sprintf("golden-scale-%dx%dx%d", w, h, ch), mk(0, w, h, ch, 0, w/2, h/2, ch, 0, 1))
			for _, d := range [][2]int{{w / 2, h / 2}, {101, 77}, {w + 60, h + 20}, {w, 1}} {
				add(fmt.Sprintf("golden-resample-%dx%dx%d-%dx%d", w, h, ch, d[0], d[1]), mk(4, w, h, ch, 0, d[0], d[1], ch, 0, 2))
			}
			for _, cw := range []bool{true, false} {
				op := byte(3)
				if cw {
					op = 2
				}
				add(fmt.Sprintf("golden-rotate-%dx%dx%d-cw%v", w, h, ch, cw), mk(op, w, h, ch, 0, h, w, ch, 0, 3))
			}
		}
		add(fmt.Sprintf("golden-luma-%dx%d", w, h), mk(1, w, h, 3, 0, w, h, 1, 0, 4))
	}

	// The four Fuzz* bodies, 200 random parameter sets each, with the Go parameter ranges.
	r := rng(0x9E3779B97F4A7C15)
	for i := 0; i < 200; i++ {
		ch, seed := 1+2*(r.next()&1), uint64(r.next())
		w, h, pad, dpad := r.next()%130+2, r.next()%6+2, r.next()%5, r.next()%3
		add(fmt.Sprintf("fuzz-scale-%03d", i), mk(0, w, h, ch, pad, w/2, h/2, ch, dpad, seed))
		w, h = w-1, h-1 // luma: w%130+1, h%6+1
		add(fmt.Sprintf("fuzz-luma-%03d", i), mk(1, w, h, 3, pad, w, h, 1, dpad, seed))
		w, h = r.next()%50+1, r.next()%50+1
		cw := r.next()&1 == 1
		op := byte(3)
		if cw {
			op = 2
		}
		add(fmt.Sprintf("fuzz-rotate-%03d", i), mk(op, w, h, ch, 1, h, w, ch, 0, seed))
		w, h = r.next()%90+1, r.next()%90+1
		add(fmt.Sprintf("fuzz-resample-%03d", i), mk(4, w, h, ch, 2, r.next()%120+1, r.next()%120+1, ch, 0, seed))
	}

	// Mutations of small valid calls: every truncation near the header and eight along the
	// pixels, every header bit flip (but dlen bits 12..20, which only enlarge the output),
	// and every length field set to edge values.
	bases := []tc{
		mk(0, 17, 7, 1, 1, 8, 3, 1, 0, 1), mk(0, 9, 5, 3, 0, 4, 2, 3, 0, 2),
		mk(1, 20, 4, 3, 2, 20, 4, 1, 1, 3), mk(2, 9, 6, 3, 1, 6, 9, 3, 0, 4),
		mk(3, 7, 5, 1, 0, 5, 7, 1, 0, 5), mk(4, 12, 9, 1, 0, 7, 5, 1, 0, 6),
		mk(4, 6, 6, 3, 2, 9, 4, 3, 0, 7),
	}
	for i, c := range bases {
		b := c.bytes()
		name := fmt.Sprintf("mut%d", i)
		add(name+"-base", c)
		cuts := map[int]bool{}
		for l := 0; l <= hdrLen+2; l++ {
			cuts[l] = true
		}
		for k := 1; k < 8; k++ {
			cuts[len(b)*k/8] = true
		}
		for _, l := range sortedKeys(cuts) {
			addRaw(fmt.Sprintf("%s-cut%04d", name, l), b[:l])
		}
		for bit := 0; bit < hdrLen*8; bit++ {
			if d := bit - 8*(5+6*4); d >= 12 && d <= 20 { // dlen is the last field
				continue
			}
			m := append([]byte(nil), b...)
			m[bit/8] ^= 1 << (bit % 8)
			addRaw(fmt.Sprintf("%s-flip%03d", name, bit), m)
		}
		for f := 0; f < 7; f++ {
			old := binary.LittleEndian.Uint32(b[5+4*f:])
			for j, v := range []uint32{0, 1, 2, 1<<31 - 1, 1<<32 - 1, old - 1, old + 1} {
				m := append([]byte(nil), b...)
				binary.LittleEndian.PutUint32(m[5+4*f:], v)
				addRaw(fmt.Sprintf("%s-field%d-%d", name, f, j), m)
			}
		}
	}
	fmt.Fprintf(os.Stderr, "generated %d inputs\n", count)
	return werr
}

func sortedKeys(m map[int]bool) []int {
	var k []int
	for v := range m {
		k = append(k, v)
	}
	sort.Ints(k)
	return k
}

// ---- benchmark -----------------------------------------------------------------------------

// benchAll prints name, destination bytes and ns per call for each kernel on a letter page at
// 300 dpi, as bench_test.go does. examples/bench.rs does the same.
func benchAll(n int) {
	const w, h = 2550, 3300
	pl := func(w, h, ch int) raster.Plane {
		return raster.Plane{Pix: page(1, w, h, ch), W: w, H: h, Stride: w * ch, Channels: ch}
	}
	ctx, lim := context.Background(), raster.Limits{}
	for _, k := range []struct {
		name             string
		sch, dw, dh, dch int
		fn               func(dst, src raster.Plane) error
	}{
		{"scale-gray", 1, w / 2, h / 2, 1, func(d, s raster.Plane) error { return raster.Scale2to1(ctx, d, s, lim) }},
		{"scale-rgb", 3, w / 2, h / 2, 3, func(d, s raster.Plane) error { return raster.Scale2to1(ctx, d, s, lim) }},
		{"luma", 3, w, h, 1, func(d, s raster.Plane) error { return raster.Luma(ctx, d, s, lim) }},
		{"resample", 1, w / 2, h / 2, 1, func(d, s raster.Plane) error { return raster.Resample(ctx, d, s, lim) }},
		{"resample-rgb", 3, w / 2, h / 2, 3, func(d, s raster.Plane) error { return raster.Resample(ctx, d, s, lim) }},
		{"rotate", 1, h, w, 1, func(d, s raster.Plane) error { return raster.Rotate90(ctx, d, s, true, lim) }},
		{"rotate-rgb", 3, h, w, 3, func(d, s raster.Plane) error { return raster.Rotate90(ctx, d, s, true, lim) }},
	} {
		src, dst := pl(w, h, k.sch), pl(k.dw, k.dh, k.dch)
		check(k.fn(dst, src))
		t := time.Now()
		for i := 0; i < n; i++ {
			check(k.fn(dst, src))
		}
		fmt.Printf("%s\t%d\t%d\n", k.name, len(dst.Pix), time.Since(t).Nanoseconds()/int64(n))
	}
}
