# uhf-mrsf

A Python prototype for aligning molecular orbitals (MOs) between alpha and beta spin channels for unrestricted MRSF (UMRSF) calculations

## 🧩 Molden format modification

To add a `Sym= C1` line before each `Ene=` line in a molden file using Vim, use the following command:

```
%g/^Ene=.*/exec "normal! OSym= C1"
```
