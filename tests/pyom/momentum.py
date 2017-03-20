from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import sys

from pyomtest import PyOMTest
from climate import Timer
from climate.pyom.core import momentum, external, numerics

class MomentumTest(PyOMTest):
    extra_settings = {
                        "coord_degree": True,
                        "enable_cyclic_x": True,
                        "enable_hydrostatic": True,
                        "enable_conserve_energy": True,
                        "enable_bottom_friction_var": True,
                        "enable_hor_friction_cos_scaling": True,
                        "enable_implicit_vert_friction": True,
                        "enable_explicit_vert_friction": True,
                        "enable_TEM_friction": True,
                        "enable_hor_friction": True,
                        "enable_biharmonic_friction": True,
                        "enable_ray_friction": True,
                        "enable_bottom_friction": True,
                        "enable_quadratic_bottom_friction": True,
                        "enable_momentum_sources": True,
                        "enable_streamfunction": True,
                        "congr_epsilon": 1e-12,
                     }
    first = True
    def initialize(self):
        m = self.pyom_legacy.main_module

        np.random.seed(123456)
        self.set_attribute("hor_friction_cosPower", np.random.randint(1,5))

        for a in ("dt_mom", "r_bot", "r_quad_bot", "A_h", "A_hbi", "AB_eps", "x_origin", "y_origin"):
            self.set_attribute(a, np.random.rand())

        for a in ("dxt",):
            self.set_attribute(a,100 * np.ones(self.nx+4) + np.random.rand(self.nx+4))

        for a in ("dyt",):
            self.set_attribute(a,100 * np.ones(self.ny+4) + np.random.rand(self.ny+4))

        for a in ("dzt",):
            self.set_attribute(a,np.random.rand(self.nz))

        for a in ("r_bot_var_u", "r_bot_var_v", "surface_taux", "surface_tauy", "coriolis_t", "coriolis_h"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4))

        for a in ("area_u", "area_v", "area_t", "hur", "hvr"):
            self.set_attribute(a,np.random.rand(self.nx+4,self.ny+4))

        for a in ("K_diss_v", "kappaM", "flux_north", "flux_east", "flux_top", "K_diss_bot", "K_diss_h",
                  "du_mix", "dv_mix", "u_source", "v_source", "du_adv", "dv_adv"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz))

        for a in ("u","v","w","du","dv"):
            self.set_attribute(a,np.random.randn(self.nx+4,self.ny+4,self.nz,3))

        kbot = np.random.randint(1, self.nz, size=(self.nx+4,self.ny+4))
        # add some islands, but avoid boundaries
        kbot[3:-3,3:-3].flat[np.random.randint(0, (self.nx-2) * (self.ny-2), size=10)] = 0
        self.set_attribute("kbot",kbot)

        if self.first:
            numerics.calc_grid(self.pyom_new)
            numerics.calc_topo(self.pyom_new)
            external.streamfunction_init(self.pyom_new)
            self.pyom_legacy.fortran.calc_grid()
            self.pyom_legacy.fortran.calc_topo()
            self.pyom_legacy.fortran.streamfunction_init()
            self.first = False

        self.test_module = momentum
        pyom_args = (self.pyom_new,)
        pyom_legacy_args = dict()
        self.test_routines = OrderedDict()
        self.test_routines["momentum_advection"] = (pyom_args, pyom_legacy_args)
        self.test_routines["vertical_velocity"] = (pyom_args, pyom_legacy_args)
        self.test_routines["momentum"] = (pyom_args, pyom_legacy_args)


    def test_passed(self,routine):
        all_passed = True
        for f in ("flux_east","flux_north","flux_top","u","v","w","K_diss_v","du_adv","dv_adv","du","dv",
                  "K_diss_bot","K_diss_h","du_mix","dv_mix","psi","dpsi","du_cor","dv_cor"):
            passed = self._check_var(f)
            if not passed:
                all_passed = False
        plt.show()
        return all_passed

    def _normalize(self,*arrays):
        norm = np.abs(arrays[0]).max()
        if norm == 0.:
            return arrays
        return (a / norm for a in arrays)

    def _check_var(self,var):
        v1, v2 = self.get_attribute(var)
        if v1.ndim > 1:
            v1 = v1[2:-2, 2:-2, ...]
        if v2.ndim > 1:
            v2 = v2[2:-2, 2:-2, ...]
        if v1 is None or v2 is None:
            raise RuntimeError(var)
        passed = np.allclose(*self._normalize(v1,v2))
        if not passed:
            print(var, np.abs(v1-v2).max(), v1.max(), v2.max(), np.where(v1 != v2))
            while v1.ndim > 2:
                v1 = v1[...,-1]
            while v2.ndim > 2:
                v2 = v2[...,-1]
            if v1.ndim == 2:
                fig, axes = plt.subplots(1,3)
                axes[0].imshow(v1)
                axes[0].set_title("New")
                axes[1].imshow(v2)
                axes[1].set_title("Legacy")
                axes[2].imshow(v1 - v2)
                axes[2].set_title("diff")
                fig.suptitle(var)
        return passed

if __name__ == "__main__":
    test = MomentumTest(80, 70, 50, fortran=sys.argv[1])
    passed = test.run()
