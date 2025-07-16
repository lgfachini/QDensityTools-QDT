import numpy as np
import re

class WFXParser:
    def __init__(self, filename):
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

        # NÚCLEOS
        coords_raw = list(map(float, self._parse_tag_block(content, "Nuclear Cartesian Coordinates").split()))
        coords = np.array(coords_raw).reshape(-1, 3)

        atomic_numbers = list(map(int, self._parse_tag_block(content, "Atomic Numbers").split()))
        nuclear_charges = list(map(float, self._parse_tag_block(content, "Nuclear Charges").split()))

        for z, charge, coord in zip(atomic_numbers, nuclear_charges, coords):
            self.data['nuclei'].append({
                'atomic_number': z,
                'charge': charge,
                'coords': coord,  # em bohr
                'symbol': self._element_symbol(z)
            })

        # PRIMITIVOS
        self.data['primitive_centers'] = list(map(int, self._parse_tag_block(content, "Primitive Centers").split()))
        self.data['primitive_types'] = list(map(int, self._parse_tag_block(content, "Primitive Types").split()))
        self.data['primitive_exponents'] = list(map(float, self._parse_tag_block(content, "Primitive Exponents").split()))
        n_prim = len(self.data['primitive_centers'])

        # ORBITAIS MOLECULARES
        spin_types_raw = self._parse_tag_block(content, "Molecular Orbital Spin Types")
        spin_types = [line.strip() for line in spin_types_raw.splitlines() if line.strip()]
        occs = list(map(float, self._parse_tag_block(content, "Molecular Orbital Occupation Numbers").split()))
        energies = list(map(float, self._parse_tag_block(content, "Molecular Orbital Energies").split()))
        mo_block = self._parse_tag_block(content, "Molecular Orbital Primitive Coefficients")

        # Divide os blocos por <MO Number>
        mo_splits = re.split(r"<\s*MO\s*Number\s*>\s*\d+\s*</\s*MO\s*Number\s*>", mo_block)
        mo_splits = [blk.strip() for blk in mo_splits if blk.strip()]

        if not (len(spin_types) == len(occs) == len(energies) == len(mo_splits)):
            raise ValueError(
                f"Inconsistência no número de orbitais: "
                f"spin_types={len(spin_types)}, ocupações={len(occs)}, energias={len(energies)}, MO blocos={len(mo_splits)}"
            )

        # Preenche orbitais
        for i, (spin, occ, en, blk) in enumerate(zip(spin_types, occs, energies, mo_splits)):
            coeffs = []
            for line in blk.splitlines():
                line = line.strip()
                if line:
                    coeffs.extend(map(float, line.split()))

            if len(coeffs) != n_prim:
                raise ValueError(
                    f"Erro no MO {i}: {len(coeffs)} coeficientes lidos, mas esperado {n_prim} primitivos."
                )

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
                raise ValueError(f"Tipo de spin desconhecido: '{spin}'")

    def _element_symbol(self, atomic_number):
        periodic_table = [
            '', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
        ]
        return periodic_table[atomic_number] if 0 < atomic_number < len(periodic_table) else 'X'
