import numpy as np
import re
from periodic_table import get_symbol_from_atomic_number
from scipy.ndimage import gaussian_filter

class WFXParser:
    """
    Parser for legacy .wfx wavefunction files.
    Extracts nuclei information, primitives, and molecular orbitals (MOs).
    """

    def __init__(self, filename):
        """
        Initialize parser and parse the file.
        
        Parameters:
            filename (str): Path to the legacy .wfx file.
        """
        self.filename = filename
        self.data = {
            'nuclei': [],
            'mo_coefficients_alpha': [],
            'mo_coefficients_beta': [],
            'mo_occupations_alpha': [],
            'mo_occupations_beta': [],
            'mo_energies_alpha': [],
            'mo_energies_beta': [],
            'primitive_centers': [],
            'primitive_types': [],
            'primitive_exponents': []
        }
        self._parse_legacy_wfx()

    def _parse_tag_block(self, content, tag):
        pattern = rf"<\s*{re.escape(tag)}\s*>\s*(.*?)\s*<\s*/\s*{re.escape(tag)}\s*>"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else None

    def _parse_legacy_wfx(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            content = f.read()

        coords_raw = list(map(float, self._parse_tag_block(content, "Nuclear Cartesian Coordinates").split()))
        coords = np.array(coords_raw).reshape(-1, 3)

        atomic_numbers = list(map(int, self._parse_tag_block(content, "Atomic Numbers").split()))
        nuclear_charges = list(map(float, self._parse_tag_block(content, "Nuclear Charges").split()))

        # Use get_symbol_from_atomic_number to assign symbol
        for z, charge, coord in zip(atomic_numbers, nuclear_charges, coords):
            symbol = get_symbol_from_atomic_number(z)
            if symbol is None:
                symbol = 'X'  # fallback
            self.data['nuclei'].append({
                'atomic_number': z,
                'charge': charge,
                'coords': coord,
                'symbol': symbol
            })

        self.data['primitive_centers'] = list(map(int, self._parse_tag_block(content, "Primitive Centers").split()))
        self.data['primitive_types'] = list(map(int, self._parse_tag_block(content, "Primitive Types").split()))
        self.data['primitive_exponents'] = list(map(float, self._parse_tag_block(content, "Primitive Exponents").split()))
        n_prim = len(self.data['primitive_centers'])

        spin_types_raw = self._parse_tag_block(content, "Molecular Orbital Spin Types")
        spin_types = [line.strip() for line in spin_types_raw.splitlines() if line.strip()]
        occs = list(map(float, self._parse_tag_block(content, "Molecular Orbital Occupation Numbers").split()))
        energies = list(map(float, self._parse_tag_block(content, "Molecular Orbital Energies").split()))
        mo_block = self._parse_tag_block(content, "Molecular Orbital Primitive Coefficients")

        mo_splits = re.split(r"<\s*MO\s*Number\s*>\s*\d+\s*</\s*MO\s*Number\s*>", mo_block)
        mo_splits = [blk.strip() for blk in mo_splits if blk.strip()]

        if not (len(spin_types) == len(occs) == len(energies) == len(mo_splits)):
            raise ValueError(
                f"Inconsistent MO data lengths: spin_types={len(spin_types)}, occs={len(occs)}, "
                f"energies={len(energies)}, mo_splits={len(mo_splits)}"
            )

        for i, (spin, occ, en, blk) in enumerate(zip(spin_types, occs, energies, mo_splits)):
            coeffs = []
            for line in blk.splitlines():
                line = line.strip()
                if line:
                    coeffs.extend(map(float, line.split()))

            if len(coeffs) != n_prim:
                raise ValueError(f"MO {i} has {len(coeffs)} coeffs but expected {n_prim}")

            spin_l = spin.lower().replace(" ", "")
            if spin_l == "alpha":
                self.data['mo_coefficients_alpha'].append(coeffs)
                self.data['mo_occupations_alpha'].append(occ)
                self.data['mo_energies_alpha'].append(en)
            elif spin_l == "beta":
                self.data['mo_coefficients_beta'].append(coeffs)
                self.data['mo_occupations_beta'].append(occ)
                self.data['mo_energies_beta'].append(en)
            elif spin_l == "alphaandbeta":
                self.data['mo_coefficients_alpha'].append(coeffs)
                self.data['mo_coefficients_beta'].append(coeffs)
                self.data['mo_occupations_alpha'].append(occ / 2.0)
                self.data['mo_occupations_beta'].append(occ / 2.0)
                self.data['mo_energies_alpha'].append(en)
                self.data['mo_energies_beta'].append(en)
            else:
                raise ValueError(f"Unknown spin type: '{spin}'")

class CubeParser:
    """
    Parser for Gaussian-style volumetric .cube files.
    """

    def __init__(self, filename, smoothing_sigma=0.0):
        self.filename = filename
        self.origin = None
        self.vectors = None
        self.grid = None
        self.atoms = []
        self.density = None
        self.data = {}
        self.smoothing_sigma = smoothing_sigma
        self._parse_cube()

    def _parse_cube(self):
        with open(self.filename, 'r') as f:
            lines = f.readlines()

        parts = lines[2].split()
        n_atoms = abs(int(parts[0]))
        self.origin = np.array(list(map(float, parts[1:4])))

        nx_line = list(map(float, lines[3].split()))
        ny_line = list(map(float, lines[4].split()))
        nz_line = list(map(float, lines[5].split()))
        nx, ny, nz = int(nx_line[0]), int(ny_line[0]), int(nz_line[0])
        self.grid = np.array([nx, ny, nz])

        vx, vy, vz = np.array(nx_line[1:]), np.array(ny_line[1:]), np.array(nz_line[1:])
        self.vectors = np.array([vx, vy, vz])

        nuclei = []
        for i in range(6, 6 + n_atoms):
            parts = lines[i].split()
            Z = int(parts[0])
            x, y, z = map(float, parts[2:5])
            symbol = get_symbol_from_atomic_number(Z)
            if symbol is None:
                symbol = 'X'
            coords = np.array([x, y, z])
            self.atoms.append((Z, x, y, z))
            nuclei.append({'Z': Z, 'coords': coords, 'symbol': symbol})

        scalar_values = []
        for line in lines[6 + n_atoms:]:
            scalar_values.extend(map(float, line.split()))

        density_array = np.array(scalar_values).reshape((nx, ny, nz))

        if self.smoothing_sigma > 0:
            density_array = gaussian_filter(density_array, sigma=self.smoothing_sigma)

        self.density = density_array

        self.data['density'] = self.density
        self.data['origin'] = self.origin
        self.data['vectors'] = self.vectors
        self.data['nuclei'] = nuclei
