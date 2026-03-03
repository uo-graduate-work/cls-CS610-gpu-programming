#!/usr/bin/env python3

from pathlib import Path


def read_pnm_payload(path: Path):
    with path.open("rb") as f:
        magic = f.readline().strip()
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()

        parts = line.split()
        if len(parts) == 2:
            width, height = map(int, parts)
            maxval = int(f.readline().strip())
        else:
            width = int(parts[0])
            height = int(f.readline().strip())
            maxval = int(f.readline().strip())

        payload = f.read()

    return magic, width, height, maxval, payload


def exact_match(a: Path, b: Path):
    return a.read_bytes() == b.read_bytes()


def payload_diff(a: Path, b: Path):
    ma, wa, ha, xa, pa = read_pnm_payload(a)
    mb, wb, hb, xb, pb = read_pnm_payload(b)
    if (ma, wa, ha, xa) != (mb, wb, hb, xb):
        return None, None, (ma, wa, ha, xa), (mb, wb, hb, xb)
    if len(pa) != len(pb):
        return None, None, len(pa), len(pb)

    diffs = 0
    max_diff = 0
    for x, y in zip(pa, pb):
        d = abs(x - y)
        if d:
            diffs += 1
            if d > max_diff:
                max_diff = d
    return diffs, max_diff, None, None


def main():
    root = Path(".")

    task02_out = root / "output.ppm"
    task02_ans = root / "answers" / "task02_correct_output.ppm"

    if not task02_out.exists():
        print("Task02: missing output.ppm (run ./task02 first)")
    else:
        exact = exact_match(task02_out, task02_ans)
        diffs, max_diff, a_meta, b_meta = payload_diff(task02_out, task02_ans)
        print("Task02:")
        print(f"  exact-byte-match: {'YES' if exact else 'NO'}")
        if diffs is not None:
            print(f"  payload-diff-pixels: {diffs}")
            print(f"  payload-max-abs-diff: {max_diff}")
        else:
            print(f"  metadata mismatch: {a_meta} vs {b_meta}")

    pairs = [
        (
            root / "out_r1.ppm",
            root / "answers" / "task03_correct_output_radius_1.ppm",
            "Task03 radius 1",
        ),
        (
            root / "out_r2.ppm",
            root / "answers" / "task03_correct_output_radius_2.ppm",
            "Task03 radius 2",
        ),
        (
            root / "out_r4.ppm",
            root / "answers" / "task03_correct_output_radius_4.ppm",
            "Task03 radius 4",
        ),
        (
            root / "out_r8.ppm",
            root / "answers" / "task03_correct_output_radius_8.ppm",
            "Task03 radius 8",
        ),
    ]

    for outp, ansp, label in pairs:
        if not outp.exists():
            print(
                f"{label}: missing {outp.name} (run task03 with --mode basic --radius N --output out_rN.ppm)"
            )
            continue

        exact = exact_match(outp, ansp)
        diffs, max_diff, a_meta, b_meta = payload_diff(outp, ansp)
        print(f"{label}:")
        print(f"  exact-byte-match: {'YES' if exact else 'NO'}")
        if diffs is not None:
            print(f"  payload-diff-bytes: {diffs}")
            print(f"  payload-max-abs-diff: {max_diff}")
        else:
            print(f"  metadata mismatch: {a_meta} vs {b_meta}")


if __name__ == "__main__":
    main()
