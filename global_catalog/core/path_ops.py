def path_pretty(l1, l2, l3) -> str:
    def s(x): return "nan" if x is None or str(x).strip() == "" else str(x)
    return f"{s(l1)} / {s(l2)} / {s(l3)}"

