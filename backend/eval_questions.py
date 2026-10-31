"""Quick quality check: runs sample student/parent questions through the searcher.

Usage (from backend/):  python eval_questions.py
Each case is (question, text expected in the answer, or None = should fall back).
"""
from search import Searcher

CASES = [
    ("where is the college", "Wadala (East)"),
    ("college address", "Wadala (East)"),
    ("how to reach vit", "bus no. 110"),
    ("contact number", "022-2416 1126"),
    ("what is the official website", "vit.edu.in"),
    ("which branches are available", "Biomedical Engineering"),
    ("what courses does vit offer", "MMS"),
    ("how do I get admission to btech", "two routes"),
    ("is admission through mht cet cap", "MHT-CET"),
    ("direct second year admission for diploma students", "Direct Second Year"),
    ("mtech admission", "M.Tech"),
    ("mms admission eligibility", "MMS"),
    ("phd admission", "research centre"),
    ("fees for btech", "2,03,000"),
    ("how much is the fee", "2,03,000"),
    ("fees for obc category", "1,14,740"),
    ("institute quota fees for computer engineering", "4,00,000"),
    ("mtech fees", "55,000"),
    ("mms fees", "1,99,500"),
    ("lateral entry fees", "1,82,000"),
    ("documents required for admission", "allotment letter"),
    ("how to pay fees online", "Cheques are NOT accepted"),
    ("refund if I cancel admission", "Rs. 1,000"),
    ("download admission brochure", "brochure"),
    ("scholarships available", "TFWS"),
    ("how are placements", "37.5 LPA"),
    ("internship opportunities", "stipend"),
    ("is vit autonomous", "Autonomous"),
    ("naac grade of vit", "A+"),
    ("who is the principal", "Sangeeta Joshi"),
    ("who is the director", "Milind Tadvalkar"),
    ("is there a hostel", "Nerul"),
    ("canteen facility", "canteen"),
    ("student clubs and fests", "VERVE"),
    ("alumni network", "Alumni"),
    ("faculty job openings", "Careers"),
    ("anti ragging committee", "ragging"),
    ("difference between computer engineering and IT", "Information Technology"),
    ("is there AI ML branch", "Honours"),
    ("college timings", "9:00 am"),
    ("weather in delhi", None),
    ("best pizza in mumbai", None),
    ("who won the cricket world cup", None),
]


def main():
    s = Searcher()
    passed = 0
    methods = {}
    for q, expect in CASES:
        r = s.answer(q)
        if expect is None:
            ok = not r["matched"]
        else:
            ok = r["matched"] and expect.lower() in r["answer"].lower()
        passed += ok
        methods[r.get("method")] = methods.get(r.get("method"), 0) + 1
        print(f"{'PASS' if ok else 'FAIL'}  {r['confidence']:.2f}  {q!r} [{r.get('method')}] -> {r['answer'][:60]!r}")
    print(f"\n{passed}/{len(CASES)} passed")
    print("Answered by:", ", ".join(f"{k}: {v}" for k, v in methods.items()))


if __name__ == "__main__":
    main()
