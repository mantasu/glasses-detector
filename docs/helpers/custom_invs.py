import os
import sys
from importlib.metadata import version

import sphobjinv as soi


class CustomInvs:
    def __init__(self, static_path: str = "_static"):
        # Init inv directory and create it if not exists
        self.inv_dir = os.path.join(static_path, "inv")
        os.makedirs(self.inv_dir, exist_ok=True)

    def create_tqdm_inv(self) -> dict[str, tuple[str, str]]:
        # Init inv and module
        inv = soi.Inventory()

        # Define the Sphinx header information
        inv.project = "tqdm"
        inv.version = version("tqdm")

        # Define the tqdm class
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
        path = os.path.join(self.inv_dir, "tqdm.inv")
        text = soi.compress(inv.data_file(contract=True))
        soi.writebytes(path, text)

        return {"tqdm": ("https://tqdm.github.io/docs/", path)}

    def create_builtin_constants_inv(self) -> dict[str, tuple[str, str]]:
        # Init inv and module
        inv = soi.Inventory()

        # Define the Sphinx header information
        inv.project = "builtin_constants"
        major, minor, micro = sys.version_info[:3]
        inv.version = f"{major}.{minor}.{micro}"

        for constant in ["None", "True", "False"]:
            # Define the constant as class
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
        path = os.path.join(self.inv_dir, "builtin_constants.inv")
        text = soi.compress(inv.data_file(contract=True))
        soi.writebytes(path, text)

        return {"builtin_constants": ("https://docs.python.org/3", path)}

    def __call__(self) -> dict[str, tuple[str, str]]:
        # Init custom invs
        custom_invs = {}

        for method_name in dir(self):
            if method_name.startswith("create_"):
                # Update custom invs dictionary with the new one
                custom_invs.update(getattr(self, method_name)())

        return custom_invs
