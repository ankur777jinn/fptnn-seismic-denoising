# Data provenance
All files come from the open repository cited in footnote 4 of Qian et al. (2022):
https://github.com/chenyk1990/reproducible_research (commit 795d2e2, GPL-2.0)

| file | origin in that repo | content |
|---|---|---|
| raw/yc_synth5d.mat | sgk5d/matfun/yc_synth5d.mat | 5-D synthetic (t,hx,hy,x,y)=100x10x10x10x10, dt=4 ms -- the exact 5-D set of the paper |
| raw/yc_hyper5d.mat | drr5d/matfun/yc_hyper5d.mat | 5-D curved-event synthetic 101x32x32x5x5 (4-D part used) |
| raw/real3d_field.bin | drr3d/matfun/real3d.bin | 3-D post-stack field data, float32, MATLAB order [300,1000] -> (t,x,y)=300x100x10 |

The paper's 4-D VSP set (footnote 5, SEG wiki "3D 9C synthetic VSP data") is no longer downloadable
(S3 returns AccessDenied), and the WY field data are proprietary.
