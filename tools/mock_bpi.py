"""LOCAL TESTING ONLY: writes a CSV with the exact BPIC-2019 column schema so the pipeline can be smoke-tested.
Never used for results."""
import numpy as np, pandas as pd, sys
rng = np.random.default_rng(0)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12000
vend = [f"vendorID_{i:04d}" for i in range(400)]
vlevel = rng.lognormal(6, 1.3, len(vend))
users = [f"user_{i:03d}" for i in range(80)]
areas = ["Packaging", "Sales", "Logistics", "Enterprise Services", "Marketing", "Products for Resale"]
rows, eid = [], 0
t0 = pd.Timestamp("2018-01-01")
for c in range(N):
    v = int(rng.integers(len(vend)))
    po = f"45{rng.integers(10**7, 10**8)}"
    case = f"{po}_{(c % 5 + 1) * 10:05d}"
    grb = rng.random() < 0.7
    area = areas[v % len(areas)]
    amt = round(float(vlevel[v] * rng.lognormal(0, 0.35)), 2)
    tpo = t0 + pd.Timedelta(days=float(rng.uniform(0, 360)))
    ua, ub, uc = rng.choice(users, 3)
    if rng.random() < 0.03: uc = ua
    base = {"case Spend area text": area, "case Company": "companyID_0000", "case Document Type": "Standard PO",
            "case Sub spend area text": "x", "case Purch. Doc. Category name": "Purchase order", "case Vendor": vend[v],
            "case Item Type": "Standard" if grb else "Service", "case Item Category": "3-way match, invoice after GR" if grb else "2-way match",
            "case Spend classification text": "NPR", "case Source": "sourceSystemID_0000", "case Name": f"vendor{v}",
            "case GR-Based Inv. Verif.": grb, "case Item": "10", "case concept:name": case, "case Goods Receipt": grb}
    ev = [("Create Purchase Order Item", ua, amt, tpo)]
    tgr = tpo + pd.Timedelta(days=float(rng.uniform(1, 20)))
    if grb and rng.random() < 0.95: ev.append(("Record Goods Receipt", ub, amt, tgr))
    if rng.random() < 0.9:
        tin = tgr + pd.Timedelta(days=float(rng.uniform(-3, 15)))
        ev.append(("Record Invoice Receipt", uc, round(amt * (1 + (rng.random() < 0.05) * rng.uniform(0, 0.3)), 2), tin))
        ev.append(("Clear Invoice", "batch_00", amt, tin + pd.Timedelta(days=30)))
    for a, u, m, t in ev:
        r = dict(base); r.update({"eventID ": eid, "event User": u, "event org:resource": u, "event concept:name": a,
                                  "event Cumulative net worth (EUR)": m, "event time:timestamp": t.strftime("%d-%m-%Y %H:%M:%S.000")})
        rows.append(r); eid += 1
cols = ["eventID ", "case Spend area text", "case Company", "case Document Type", "case Sub spend area text",
        "case Purch. Doc. Category name", "case Vendor", "case Item Type", "case Item Category", "case Spend classification text",
        "case Source", "case Name", "case GR-Based Inv. Verif.", "case Item", "case concept:name", "case Goods Receipt",
        "event User", "event org:resource", "event concept:name", "event Cumulative net worth (EUR)", "event time:timestamp"]
pd.DataFrame(rows)[cols].to_csv(sys.argv[1], index=False, encoding="latin-1")
print("wrote", len(rows), "events")
