# Run report: Kalman Filters

- document compiled: **True**, pages: 9, overfull boxes: 0, undefined refs: 0
- words: 2255 (budget 2640, ratio 0.854)
- figures: 8/8 rendered, mean key-point coverage 1.0
- autofixes: 22, quarantined units: 0, plan fallback: False
- time: 203.2 s

| section | words / target | coverage | figures | reward | steps |
|---|---|---|---|---|---|
| s1 Introduction to Kalman Filters | 420 / 400 | 1.0 | 1/1 | 0.706 | draft > repair_figure > revise |
| s2 Linear Dynamic Systems | 432 / 500 | 1.0 | 2/2 | 0.849 | draft > repair_figure > repair_figure > revise |
| s3 Kalman Filter Equations | 457 / 600 | 1.0 | 2/2 | 0.816 | draft > repair_figure > repair_figure > revise |
| s4 Example: Kalman Filter for a Simple Syst | 446 / 600 | 1.0 | 2/2 | 0.811 | draft > repair_figure > revise |
| s5 Summary and Applications | 500 / 540 | 1.0 | 1/1 | 0.746 | draft > repair_figure |

Cheat sheet: 1 page(s) (budget 1), layout [3, 'footnotesize'], reward 0.75

LLM usage:

- plan: 1 calls, 498 prompt / 1119 completion tokens, 22.7s
- write_section: 15 calls, 13764 prompt / 15278 completion tokens, 586.9s
- repair_figure: 7 calls, 3452 prompt / 1444 completion tokens, 45.7s
- revise_section: 4 calls, 6894 prompt / 4355 completion tokens, 117.4s
- cheatsheet: 1 calls, 675 prompt / 906 completion tokens, 19.2s
