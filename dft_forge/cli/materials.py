from __future__ import annotations

from ._helpers import _success_response


def cmd_materials_list(args: argparse.Namespace) -> int:
    from dft_forge.catalog import list_materials, get_material_profile

    names = list_materials()
    items = []
    for name in names:
        try:
            profile = get_material_profile(name)
            items.append({
                "material": name,
                "species": profile.get("species", []),
                "pseudos": profile.get("pseudos", {}),
                "ibrav": profile.get("ibrav"),
                "lattice_constant_angstrom": profile.get("lattice_constant_angstrom"),
                "input_dft": profile.get("input_dft"),
            })
        except Exception:
            items.append({"material": name})

    data = {
        "command": "materials.list",
        "count": len(items),
        "materials": items,
    }
    return _success_response(data, args.output)
