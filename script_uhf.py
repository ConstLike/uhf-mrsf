import numpy as np
from pyscf import tools


def read_molden_file(molden_file, data):
    r"""Read the Molden file and extract MO coefficients, occupations, and energies.
    This function loads data from the Molden file using PySCF
    and stores it in the provided data dictionary.
    It extracts alpha and beta MO coefficients \(C^\alpha\) and \(C^\beta\)
    (each \(\mathbb{R}^{M \times N}\)), occupation numbers, energies, and
    the AO overlap matrix \(S_{\mu\nu} = \langle \chi_\mu | \chi_\nu \rangle\).
    """
    # Load the Molden file using PySCF's tools module.
    mol, mo_energy, mo_coeff, mo_occ, _, _ = tools.molden.load(molden_file)

    # Check if the coefficients are for unrestricted orbitals (a tuple of two matrices).
    if not isinstance(mo_coeff, tuple) or len(mo_coeff) != 2:
        raise ValueError(
            "The file must contain unrestricted alpha and beta orbitals."
        )

    # Assign alpha and beta coefficients to the dictionary.
    data["MOs_alpha"], data["MOs_beta"] = mo_coeff

    # Assign alpha and beta occupation numbers to the dictionary.
    data["occ_alpha"], data["occ_beta"] = mo_occ

    # Assign alpha and beta energies to the dictionary.
    data["MOs_energy_alpha"], data["MOs_energy_beta"] = mo_energy

    # Compute and assign the AO overlap matrix to the dictionary (symmetric integral).
    data["overlap"] = mol.intor_symmetric("int1e_ovlp")

    # Number of basis functions (MOs).
    data["nbf"] = data["MOs_alpha"].shape[1]

    # Number of alpha/beta occupied MOs
    # Find occupied indices for alpha and beta.
    data["nocc_a"] = len(np.where(data["occ_alpha"] > 0.5)[0])
    data["nocc_b"] = len(np.where(data["occ_beta"] > 0.5)[0])

    # Number of alpha virtual MOs.
    data["virt_a"] = data["nbf"] - data["nocc_a"]

    # Number of beta virtual MOs.
    data["virt_b"] = data["nbf"] - data["nocc_b"]

    # Log successful loading.
    print(
        "Molden file loaded successfully. Data stored in 'data[]' dictionary."
    )


def compute_overlap_matrix(data):
    r"""Compute the overlap matrix \( M = (C^\alpha)^T S C^\beta \).
    This function calculates the overlaps between all alpha and beta orbitals in the MO basis,
    where \( M_{pq} = \langle \phi_p^\alpha | \phi_q^\beta \rangle \)."""
    # Retrieve MO coefficients and AO overlap from the dictionary.
    c_a = data["MOs_alpha"]
    c_b = data["MOs_beta"]
    s = data["overlap"]

    # Compute the MOM.
    mom = c_a.T @ s @ c_b

    return mom


def compute_l_mom(data, m_key):
    r"""Compute \( \mathcal{L} = \sum_i |MOM_{ii}|^2 \).
    This is the objective function summing squared intra-orbital overlaps,
    using the MOM from the dictionary.
    """
    # Retrieve the specified MOM from the dictionary.
    mom = data[m_key]

    # Extract the diagonal elements \( MOM_{ii} \).
    diag = np.diag(mom)

    # Sum the squares of their absolute values.
    l_mom = np.sum(np.abs(diag) ** 2).real

    return l_mom


def get_segments(data):
    r"""Identify the orbital segments based on occupations.
    This function determines HOMO and LUMO indices and
    defines two segments for alpha and beta orbitals.
    HOMO_alpha = HOMO_beta + 2
    LUMO_alpha = LUMO_beta + 2
    Segment1: Alpha from 0 to HOMO_alpha-1; Beta from 0 to LUMO_beta+1 (Alpha reordered).
    Segment2: Alpha from HOMO_alpha to last; Beta from LUMO_beta+2 to last (Beta reordered).
    Segments avoid mixing occupied and virtual to preserve reference energy.
    """
    # Retrieve number of basis functions and occupations.
    nbf = data["nbf"]
    nocc_a = data["nocc_a"]
    nocc_b = data["nocc_b"]

    # Segment 1: Alpha occupied up to HOMO-1 (excluding second SOMO).
    data["seg1_alpha"] = np.arange(0, nocc_a - 1)

    # Segment 2: Alpha from HOMO to end (virtuals, including second SOMO).
    data["seg2_alpha"] = np.arange(nocc_a, nbf)

    # Segment 1: Beta occupied up to HOMO+1 (including first SOMO).
    data["seg1_beta"] = np.arange(0, nocc_b + 1)

    # Segment 2: Beta from HOMO+2 to end (skipping first SOMO).
    data["seg2_beta"] = np.arange(nocc_b + 2, nbf)

    # Validate segment sizes, should have the same.
    print("Segments defined:")
    print(
        f"Seg1 Alpha {len(data['seg1_alpha'])}, Beta {len(data['seg1_beta'])}."
    )
    print(
        f"Seg2 Alpha {len(data['seg2_alpha'])}, Beta {len(data['seg2_beta'])}."
    )


def align_orbitals_based_on_mom(
    data, movable_indices, fixed_indices, movable_key
):
    r"""Align movable orbitals to fixed based on maximum overlaps.
    Reorders columns of movable_key ('MOs_alpha' or 'MOs_beta') to match fixed via max |MOM[i,j]|.
    Simplifies logic by always aligning movable to fixed,
    with movable_key specifying which set to reorder.
    Updates the dictionary and logs pairings."""
    # Retrieve current MOM (initial for base).
    mom = data["MOM_init"]

    # Determine if movable is alpha (rows in MOM) or beta (cols).
    is_movable_alpha = movable_key == "MOs_alpha"

    # Extract sub-MOM: rows always alpha, cols beta, but adjust based on movable.
    if is_movable_alpha:
        # Movable alpha (rows), fixed beta (cols).
        sub_mom = mom[np.ix_(movable_indices, fixed_indices)]
    else:
        # Movable beta (cols), fixed alpha (rows).
        sub_mom = mom[np.ix_(fixed_indices, movable_indices)]

    # List of available movable relative indices.
    available = list(range(len(movable_indices)))

    # List to store pairings (movable, fixed, overlap).
    pairings = []

    # Alignment loop: For each fixed, find best available movable.
    for fixed_rel, fixed_global in enumerate(fixed_indices):
        # Overlaps for fixed with available movables.
        if is_movable_alpha:
            # Fixed beta col, movable alpha rows: overlaps = sub_MOM[available, fixed_rel].
            overlaps = sub_mom[available, fixed_rel]
        else:
            # Fixed alpha row, movable beta cols: overlaps = sub_MOM[fixed_rel, available].
            overlaps = sub_mom[fixed_rel, available]

        # Find max absolute overlap.
        abs_overlaps = np.abs(overlaps)

        # Index of max overlap within available.
        max_rel_idx = np.argmax(abs_overlaps)
        max_ov = overlaps[max_rel_idx]

        # Map to global movable index.
        movable_rel = available[max_rel_idx]
        movable_global = movable_indices[movable_rel]

        # Store pairing.
        pairings.append((movable_global, fixed_global, max_ov))

        # Flip sign of movable orbital if overlap negative (to make diagonal positive).
        if max_ov < 0.0:
            data[movable_key][:, movable_global] *= -1.0
            print(
                f"Flipped sign for movable orbital {movable_global + 1} (overlap {max_ov:.6f})."
            )

        # Remove used movable from available.
        available.pop(max_rel_idx)

    # Reorder movable columns to match fixed order.
    # Sort pairings by fixed_global for alignment.
    pairings_sorted = sorted(pairings, key=lambda p: p[1])
    new_order = [p[0] for p in pairings_sorted]

    # Apply permutation to movable columns.
    data[movable_key][:, movable_indices] = data[movable_key][:, new_order]

    # Log pairings with 1-based indices.
    print("Alignment completed. Pairings (movable -> fixed, overlap):")
    for mov, fix, ov in pairings:
        print(f"Orbital {mov + 1} -> {fix + 1}, overlap = {ov:.6f}")


def optimize_jacobi_segment(data, indices, movable_key):
    r"""Optimize orbital rotations using Jacobi method within a segment.
    Performs iterative pairwise rotations on the movable orbitals (alpha or beta)
    to maximize the sum of squared diagonal overlaps in the specified indices.
    Updates the MO coefficients in place and logs convergence.
    """
    if len(indices) < 2:
        print("Segment too small for Jacobi optimization.")
        return

    # Retrieve movable and fixed coefficients, AO overlap.
    c_mov = data[movable_key]
    if movable_key == "MOs_alpha":
        c_fix = data["MOs_beta"]
    else:
        c_fix = data["MOs_alpha"]
    s = data["overlap"]

    # Convergence parameters.
    threshold = 1e-6
    max_iter = 500  # Increased for better convergence in virtual segments
    prev_l = compute_l_mom(
        data, "MOM_current"
    )  # Assume MOM_current is updated before call

    # Iterative sweeps over all pairs in the segment.
    for it in range(max_iter):
        max_theta = 0.0
        for p in range(len(indices)):
            for q in range(p + 1, len(indices)):
                i = indices[p]
                j = indices[q]

                # Extract column vectors for the pair.
                ci_mov = c_mov[:, i]
                cj_mov = c_mov[:, j]
                ci_fix = c_fix[:, i]
                cj_fix = c_fix[:, j]

                # Compute fundamental overlaps A, B, C, D for the pair.
                A = np.dot(ci_mov.T, np.dot(s, ci_fix))
                B = np.dot(cj_mov.T, np.dot(s, cj_fix))
                C = np.dot(ci_mov.T, np.dot(s, cj_fix))
                D = np.dot(cj_mov.T, np.dot(s, ci_fix))

                # Parameters for optimal angle.
                a = 0.5 * (A**2 + B**2 - C**2 - D**2)
                if movable_key == "MOs_alpha":
                    b = A * D - B * C
                else:
                    b = A * C - B * D

                # Skip if no rotation needed (numerically zero).
                if abs(a) < 1e-14 and abs(b) < 1e-14:
                    continue

                # Compute rotation angle.
                phi = np.arctan2(b, a)
                theta = 0.5 * phi
                max_theta = max(max_theta, abs(theta))

                # Apply rotation to update coefficients.
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)

                c_i_new = cos_t * ci_mov + sin_t * cj_mov
                c_j_new = -sin_t * ci_mov + cos_t * cj_mov

                c_mov[:, i] = c_i_new
                c_mov[:, j] = c_j_new

        # Monitor global L after full sweep to check improvement.
        data["MOM_temp"] = compute_overlap_matrix(data)
        current_l = compute_l_mom(data, "MOM_temp")
        if current_l < prev_l or max_theta < threshold:
            if current_l < prev_l:
                print(
                    f"Stopping early: L decreased from {prev_l:.6f} to {current_l:.6f}"
                )
            else:
                print(
                    f"Jacobi converged in {it + 1} iterations, max_theta={max_theta:.6e}"
                )
            break
        prev_l = current_l
        print(
            f"Iteration {it + 1}: L = {current_l:.6f}, max_theta={max_theta:.6e}"
        )

    else:
        print(
            f"Jacobi not converged after {max_iter} iterations, max_theta={max_theta:.6e}"
        )


def print_overlap_table(phase, data, m_key, ev=27.211386):
    r"""Print a table similar to the Fortran output for overlaps.
    This shows for each alpha orbital (occupied and virtual): index, loc of max overlap beta, energies, diagonal overlap, max overlap,
    with flags for HOMO, SOMO, rearranged, WARNING. Uses all orbitals for comprehensive testing.
    """
    # Retrieve MOM, energies, occupations from dictionary.
    M = data[m_key]
    e_a = data["MOs_energy_alpha"]
    e_b = data["MOs_energy_beta"]
    occ_a = data["occ_alpha"]
    occ_b = data["occ_beta"]

    # All alpha indices (0 to N-1).
    all_indices_a = np.arange(data["nbf"])

    # Find HOMO (last occupied alpha), SOMO (extra unoccupied beta, last 'diff' occupied alpha).
    occupied_a = np.where(occ_a > 0.5)[0]
    occupied_b = np.where(occ_b > 0.5)[0]
    num_occ_b = len(occupied_b)
    diff = len(occupied_a) - num_occ_b
    somo_indices = occupied_a[-diff:] if diff > 0 else []

    print(f"{phase}")
    print("---------------------------------------------------------")
    print("Check sign of maximum overlap between alpha and beta MOs")
    print("---------------------------------------------------------")
    print("A_i <- B   A_i, eV   B_i, eV   B_i A_i Overlap   Max Overlap")

    # Loop over all orbitals to print details.
    for k, i in enumerate(all_indices_a):
        # Overlaps for this alpha i with all beta j.
        overlaps = M[i, :]

        # Absolute values.
        abs_overlaps = np.abs(overlaps)

        # Beta index with maximum absolute overlap (1-based in print).
        loc = np.argmax(abs_overlaps)

        # Signed max overlap.
        max_ov = overlaps[loc]

        # Diagonal overlap.
        diag_ov = M[i, i]

        # Print the line (1-based indices).
        print(
            f"{k + 1:3} {loc + 1:4} {e_a[i] * ev:9.3f} {e_b[i] * ev:9.3f} {diag_ov:12.6f} {max_ov:12.6f}",
            end="",
        )

        # Flag for SOMO (in extra occupied alpha).
        if i in somo_indices:
            print("  SOMO", end="")

        # Check for rearranged or warning.
        tmp_abs = abs(max_ov)
        if i != loc and tmp_abs < 0.9:
            print("  rearranged, WARNING")
        elif i == loc and tmp_abs < 0.9:
            print("  WARNING")
        elif i != loc and tmp_abs > 0.9:
            print("  rearranged")
        else:
            print("")

    print("")


def find_low_overlap_indices(data, m_key, threshold=0.9):
    """Find indices where absolute diagonal overlap < threshold."""
    mom = data[m_key]
    diag = np.abs(np.diag(mom))
    return np.where(diag < threshold)[0]


def main(molden_file):
    r"""Main function: load data, define segments, align orbitals, and verify.
    This orchestrates the base implementation, using a dictionary for data management.
    Starts with loading, computes initial MOM and L_MOM,
    defines segments, aligns, recomputes MOM and L, and outputs diagnostics.
    """
    # Initialize data dictionary.
    data = {}

    # Load Molden file into dictionary.
    print("Stage: Loading Molden file...")
    read_molden_file(molden_file, data)

    # Compute initial MOM and store.
    print("Stage: Computing initial MOM...")
    data["MOM_init"] = compute_overlap_matrix(data)

    # Print initial overlap table for all orbitals.
    print("Stage: Printing initial overlap table...")
    print_overlap_table("INITIAL", data, "MOM_init")

    # Define segments and store in dictionary.
    print("Stage: Defining segments...")
    get_segments(data)

    # 1. Maximum Overlap Method:
    if True:
        # Align Segment 1: alpha to beta.
        print("Stage: Aligning Segment 1 (alpha to beta)...")
        align_orbitals_based_on_mom(
            data, data["seg1_alpha"], data["seg1_beta"], "MOs_alpha"
        )

        # Align Segment 2: beta to alpha.
        print("Stage: Aligning Segment 2 (beta to alpha)...")
        align_orbitals_based_on_mom(
            data, data["seg2_beta"], data["seg2_alpha"], "MOs_beta"
        )

        # Recompute MOM after alignment.
        print("Stage: Recomputing MOM after alignment...")
        data["MOM_aligned"] = compute_overlap_matrix(data)

        # Print initial overlap table for all orbitals.
        print("Stage: Printing initial overlap table...")
        print_overlap_table("ALIGNED", data, "MOM_aligned")

    # 2. Jacobi:

    if False:
        # Compute initial L and store.
        data["L_MOM_init"] = compute_l_mom(data, "MOM_init")
        print(f"Initial L_MOM: {data['L_MOM_init']:.6f}")

        # Set current for Jacobi monitoring
        data["MOM_current"] = data["MOM_init"].copy()

        # Align Segment 1: alpha to beta using Jacobi.
        print("Stage: Optimizing Segment 1 (alpha) with Jacobi...")
        optimize_jacobi_segment(data, data["seg1_alpha"], "MOs_alpha")

        # Align Segment 2: beta to alpha using Jacobi.
        print("Stage: Optimizing Segment 2 (beta) with Jacobi...")
        optimize_jacobi_segment(data, data["seg2_beta"], "MOs_beta")

        # Recompute final MOM after all rotations.
        data["MOM_after_jacobi"] = compute_overlap_matrix(data)

        # Recompute L after alignment.
        data["L_aligned"] = compute_l_mom(data, "MOM_after_jacobi")
        print(f"Aligned L: {data['L_aligned']:.6f}")

        # Print aligned overlap table for all orbitals.
        print("Stage: Printing aligned overlap table...")
        print_overlap_table("ALIGNED", data, "MOM_after_jacobi")


if __name__ == "__main__":
    MOLDEN_FILE = "thymine_ci21_scf_uhf_bhhlyp_6-31gs.molden"
    MOLDEN_FILE = "thymine_fc_scf_uhf_bhhlyp_6-31gs.molden"
    MOLDEN_FILE = "acetone_scf_uhf_bhhlyp_6-31gs.molden"
    main(MOLDEN_FILE)
