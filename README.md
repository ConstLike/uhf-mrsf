# uhf-mrsf

A Python prototype for aligning molecular orbitals (MOs) between alpha and beta spin channels for unrestricted MRSF (UMRSF) calculations

## 🧩 Molden format modification

To add a `Sym= C1` line before each `Ene=` line in a molden file using Vim, use the following command:

```
%g/^Ene=.*/exec "normal! OSym= C1"
```

## 🧪 Example Output

Running the script with a sample Molden file:

```bash
python script_uhf.py
```

Typical output:

```
Stage: Loading Molden file...
Molden file loaded successfully. Data stored in 'data[]' dictionary.
Stage: Computing initial MOM...
Stage: Printing initial overlap table...
INITIAL
---------------------------------------------------------
Check sign of maximum overlap between alpha and beta MOs
---------------------------------------------------------
A_i <- B   A_i, eV   B_i, eV   B_i A_i Overlap   Max Overlap
  1    1  -287.938  -287.497     0.999998     0.999998
  2    2  -287.578  -287.146    -0.999998    -0.999998
  ...
  38   38   119.600   120.348    -0.999991    -0.999991

Stage: Defining segments...
Segments defined:
Seg1 Alpha 8, Beta 8.
Seg2 Alpha 29, Beta 29.

Stage: Aligning Segment 1 (alpha to beta)...
Flipped sign for movable orbital 2 (overlap -0.999998).
Flipped sign for movable orbital 4 (overlap -0.998048).
...

Stage: Recomputing MOM after alignment...
Stage: Printing initial overlap table...
ALIGNED
---------------------------------------------------------
Check sign of maximum overlap between alpha and beta MOs
---------------------------------------------------------
A_i <- B   A_i, eV   B_i, eV   B_i A_i Overlap   Max Overlap
  1    1  -287.938  -287.497     0.999998     0.999998
  ...
  38   38   119.600   120.348     0.999991     0.999991
```

Lines with `WARNING` indicate special cases where overlap is less that 90%.

