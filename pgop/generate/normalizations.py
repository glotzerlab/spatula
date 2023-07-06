import os

import freud
import h5py
import numpy as np
import pgop
import tqdm


POINT_GROUPS = [
    f"C{i}" for i in range(2, 13)] + [
        f"D{i}" for i in range(2, 13)] + ["Ci", "T", "Th", "O", "Oh", "I", "Ih"]


KAPPAS = np.linspace(5.0, 25.0, 40)

N_NEIGHBORS = range(2, 21)

N_PARTICLES = 5_000


def set_diff(a, b):
    return [i for i in a if i not in b]


class StoreNorms:
    def __init__(self, name, point_groups, kappas):
        if os.path.exists:
            self._fh = h5py.File(name, "a")
        else:
            self._fh = h5py.File(name, "w")
            self._fh.attrs["kappas"] = kappas
        self._pg = point_groups

    def focus(self, neighs, kappa_idx):
        self._key = f"{neighs}/{kappa_idx}"
        if self._key not in self._fh:
            existing_pg = []
        else:
            existing_pg = self._fh[self._key].attrs.get("point_groups", [])
        self._cpg = set_diff(self._pg, existing_pg)

    def __del__(self):
        if (fh := getattr(self, "_fh", None)) is not None:
            fh.close()

    def set(self, data):
        if data.shape[1] != len(self._cpg):
            raise ValueError(f"Expected shape (-1, {len(self._cpg)})")
        if len(self._cpg) != len(self._pg):  # guarentees existence of dataset.
            ds = self._resize(data)
        else:
            ds = self._create_dataset(data)

    def _resize(self, new_data):
        orig_data = self._fh[self._key][:]
        orig_pg = self._fh[self._key].attrs["point_groups"]
        del self._fh[self._key]
        ds = self._fh.create_dataset(
            self._key, (N_PARTICLES, len(self._pg)), dtype="f4")
        ds[:, :-len(self._cpg)] = orig_data
        ds[:, -len(self._cpg)] = new_data
        ds.attrs["point_groups"] = list(orig_pg) + self._cpg
        return ds

    def _create_dataset(self, data):
        ds = self._fh.create_dataset(self._key, data=data, dtype="f4")
        ds.attrs["point_groups"] = list(self._pg)
        return ds

    @property
    def compute_point_groups(self):
        return self._cpg
        

def main(force=False):
    progress_bar = tqdm.tqdm(total=(len(N_NEIGHBORS) + 1) * len(KAPPAS))
    opt = pgop.optimize.Union.with_line_search(pgop.optimize.Mesh.from_grid())
    system = freud.data.make_random_system(
        box_size=10, num_points=N_PARTICLES, seed=389587)
    h5_store = StoreNorms("norms.h5", POINT_GROUPS, KAPPAS)
    neigh_query = freud.locality.AABBQuery.from_system(system)
    for i, n_neighs in enumerate(N_NEIGHBORS):
        nlist = neigh_query.query(
            system[1],
            {"num_neighbors": n_neighs, "exclude_ii": True}).toNeighborList()
        for j, kappa in enumerate(KAPPAS):
            h5_store.focus(n_neighs, j)
            if len(h5_store.compute_point_groups) == 0:
                progress_bar.update()
                continue
            compute = pgop.PGOP(
                "fisher", h5_store.compute_point_groups, opt, kappa=kappa)
            compute.compute(system, nlist, m=7)
            h5_store.set(compute.pgop)
            progress_bar.update()
    nlist = freud.locality.Voronoi().compute(system).toNeighborList()
    for j, kappa in enumerate(KAPPAS):
        h5_store.focus("voronoi", j)
        if len(h5_store.compute_point_groups) == 0:
            continue
        compute = pgop.PGOP(
            "fisher", h5_store.compute_point_groups, opt, kappa=kappa)
        compute.compute(system, nlist, m=7)
        h5_store.set(compute.pgop)
        progress_bar.update()
    progress_bar.close()


if __name__ == "__main__":
    print("Beginning calculation of normlization values...")
    main()
