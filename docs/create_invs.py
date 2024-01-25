import sys
from importlib.metadata import version

import sphobjinv as soi


def create_tqdm_inv():
    # Init inv and module
    inv = soi.Inventory()

    # Define the Sphinx header information
    inv.project = "tqdm"
    inv.version = version("tqdm")

    inv_obj = soi.DataObjStr(
        name="tqdm.tqdm",
        domain="py",
        role="class",
        priority="1",
        uri="tqdm/#tqdm-objects",
        dispname="tqdm",
    )
    inv.objects.append(inv_obj)

    # Write the inventory to a file
    text = inv.data_file(contract=True)
    ztext = soi.compress(text)
    soi.writebytes("_static/inv/tqdm.inv", ztext)


def create_builtin_constants_inv():
    # Init inv and module
    inv = soi.Inventory()

    # Define the Sphinx header information
    inv.project = "builtin_constants"
    major, minor, micro = sys.version_info[:3]
    inv.version = f"{major}.{minor}.{micro}"

    for constant in ["None", "True", "False"]:
        inv_obj = soi.DataObjStr(
            name=f"{constant}",
            domain="py",
            role="class",  # dummy class for linking
            priority="1",
            uri=f"library/constants.html#{constant}",
            dispname=constant,
        )
        inv.objects.append(inv_obj)

    # Write the inventory to a file
    text = inv.data_file(contract=True)
    ztext = soi.compress(text)
    soi.writebytes("_static/inv/builtin_constants.inv", ztext)


if __name__ == "__main__":
    create_tqdm_inv()
    create_builtin_constants_inv()
