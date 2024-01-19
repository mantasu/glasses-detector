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


def create_overload_inv():
    # Init inv and module
    inv = soi.Inventory()

    # Define the Sphinx header information
    inv.project = "overloads"
    major, minor, micro = sys.version_info[:3]
    inv.version = f"{major}.{minor}.{micro}"

    inv.objects.append(
        soi.DataObjStr(
            name=f"Image.Image",
            domain="py",
            role="class",
            priority="1",
            uri="https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image",
            dispname="Image",
        )
    )

    inv.objects.append(
        soi.DataObjStr(
            name=f"np.ndarray",
            domain="py",
            role="class",
            priority="1",
            uri="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html",
            dispname="ndarray",
        )
    )

    inv.objects.append(
        soi.DataObjStr(
            name=f"Collection",
            domain="py",
            role="class",
            priority="1",
            uri="https://docs.python.org/3/library/typing.html#typing.Collection",
            dispname="Collection",
        )
    )

    inv.objects.append(
        soi.DataObjStr(
            name=f"Callable",
            domain="py",
            role="class",
            priority="1",
            uri="https://docs.python.org/3/library/typing.html#typing.Callable",
            dispname="Callable",
        )
    )

    # Write the inventory to a file
    text = inv.data_file(contract=True)
    ztext = soi.compress(text)
    soi.writebytes("_static/inv/overloads.inv", ztext)


if __name__ == "__main__":
    create_tqdm_inv()
    create_builtin_constants_inv()

    # TODO: Remove when sphinx fixes overloading annotations
    create_overload_inv()
