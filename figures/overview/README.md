# Uni-LeWM method overview candidates

Prepared with academic-figure-skills (fig-analyze / fig-draw), grounded in model.py, losses.py and planner.py.

| Style | Paper asset | Editable source | Preview |
| --- | --- | --- | --- |
| Illustrated Modular | [Vector PDF](paper/uni-lewm-illustrated.pdf) | [SVG](paper/uni-lewm-illustrated.svg) | [PNG](paper/uni-lewm-illustrated.png) |
| Modern Pastel Airy — recommended | [Vector PDF](paper/uni-lewm-pastel.pdf) | [SVG](paper/uni-lewm-pastel.svg) | [PNG](paper/uni-lewm-pastel.png) |
| Dual-Fidelity Loop visual grammar | [PDF, raster](paper/uni-lewm-dual-loop.pdf) | Raster artwork | [PNG](paper/uni-lewm-dual-loop.png) |

The third style borrows the warm/cool observed-versus-imagined loop layout. Uni-LeWM uses categorical CEM; the diagram adds no Bayesian optimization or fidelity surrogate.

Use [overview.tex](paper/overview.tex) with [caption.tex](paper/caption.tex) in Overleaf. The recommended SVG/PDF preserves editable text and exact connectors. Include graphicx in the manuscript preamble. PDF assets are designed for 180 mm full-width placement; check final font size in the venue template.

The native illustrated and pastel generations are retained locally as style drafts with their failed audits. Their remaining connector errors were resolved in new deterministic vector diagrams; they are not included as paper assets. Dual-loop uses a native raster and retains a minor occluded state-history segment.

No trained policy, pixel decoder, EMA target, or goal-conditioned dynamics is claimed. The goal embedding only enters distance scoring and the observed-state STOP test. Multi-step prediction loss and SIGReg use joint encoder/dynamics training.
