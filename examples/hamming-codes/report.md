# Run report: Error-Correcting Hamming Codes

- document compiled: **True**, pages: 11, overfull boxes: 0, undefined refs: 0
- words: 2552 (budget 2640, ratio 0.967)
- figures: 11/11 rendered, mean key-point coverage 0.945
- autofixes: 0, quarantined units: 0, plan fallback: False
- time: 342.4 s

| section | words / target | coverage | figures | reward | steps |
|---|---|---|---|---|---|
| s1 Introduction to Error-Correcting Codes | 348 / 350 | 1.0 | 2/2 | 0.848 | draft > repair_figure > pick_image > revise |
| s2 Hamming Codes: A Closer Look | 443 / 430 | 0.667 | 2/2 | 0.805 | draft > repair_figure > revise |
| s3 Constructing Hamming Codes | 406 / 520 | 1.0 | 2/2 | 0.872 | draft > repair_figure > revise |
| s4 Applications of Hamming Codes | 416 / 430 | 1.0 | 2/2 | 0.887 | draft > repair_figure > repair_figure > revise |
| s5 Limitations and Efficiency of Hamming Co | 568 / 520 | 1.0 | 2/2 | 0.968 | draft > ensure_figures |
| s6 Summary and Review | 371 / 380 | 1.0 | 1/1 | 0.753 | draft > repair_figure > revise |

Cheat sheet: 1 page(s) (budget 2), layout [3, 'footnotesize'], reward 0.869

LLM usage:

- plan: 1 calls, 502 prompt / 1275 completion tokens, 32.5s
- write_section: 18 calls, 17445 prompt / 20379 completion tokens, 802.4s
- repair_figure: 5 calls, 3753 prompt / 2069 completion tokens, 73.9s
- pick_image: 3 calls, 3254 prompt / 150 completion tokens, 7.1s
- web_query: 3 calls, 1523 prompt / 15 completion tokens, 1.1s
- revise_section: 5 calls, 8590 prompt / 5541 completion tokens, 183.6s
- figure: 1 calls, 817 prompt / 364 completion tokens, 17.2s
- cheatsheet: 1 calls, 512 prompt / 1051 completion tokens, 37.4s
