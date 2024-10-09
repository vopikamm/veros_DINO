#!/usr/bin/env python

__VEROS_VERSION__ = '0+untagged.1774.g4039f76'

if __name__ == "__main__":
    raise RuntimeError(
        "Veros setups cannot be executed directly. "
        f"Try `veros run {__file__}` instead."
    )

# -- end of auto-generated header, original file below --


from veros import VerosSetup, veros_routine
from veros.variables import allocate, Variable
from veros.distributed import global_min, global_max
from veros.core.operators import numpy as npx, update, at


class ACCSetup(VerosSetup):
    """
    DINO-configuration.
    """

    @veros_routine
    def set_parameter(self, state):
        settings = state.settings
        settings.identifier = "acc2"
        settings.description = "My DINO setup"

        settings.Lx     = 50.0
        settings.Ly     = 140.0
        settings.dxy    = 1.0

        settings.nx     = settings.Lx / settings.dxy + 2 # TODO: do I need to include the boundaries?
        settings.ny     = self.mercator_grid(settings.Ly / 2.0, settings) * 2 - 1
        settings.nz = 36
        
        # TODO check time stepping
        # time in seconds
        settings.dt_mom = 2700
        settings.dt_tracer = 2700 #10800         # TODO: Can be increased? Goes unstable later, but goes unstable
        settings.runlen = 86400 * 20       # 1 year

        settings.x_origin = 0.0
        settings.y_origin = - settings.Ly / 2

        settings.coord_degree = True
        settings.enable_cyclic_x = True

        # TODO space dependent diffusion params
        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 50.0         # TODO: Changed to global flexible settings, necessary? 
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.01
        settings.enable_skew_diffusion = True

        settings.enable_hor_friction = True
        settings.A_h = (2 * settings.degtom) ** 3 * 2e-11
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1

        settings.enable_bottom_friction = True
        settings.r_bot = 1e-5

        settings.enable_implicit_vert_friction = True

        #TODO check parameters + TKE equations
        settings.enable_tke = True
        settings.c_k = 0.1
        settings.c_eps = 0.7
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True

        settings.K_gm_0 = 1000.0
        settings.enable_eke = True
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.0
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = True
        settings.enable_eke_isopycnal_diffusion = True

        settings.enable_idemix = False

        #TODO implement Roquet et at 2015
        settings.eq_of_state_type = 3

        var_meta = state.var_meta
        var_meta.update(
            t_star=Variable("t_star", ("yt",), "deg C", "Reference surface temperature"),
            t_rest=Variable("t_rest", ("xt", "yt"), "1/s", "Surface temperature restoring time scale"),
        )

    @veros_routine
    def set_grid(self, state):
        # Horizontal mercator grid
        vs = state.variables

        vs.dxt = update(vs.dxt, at[...], state.settings.dxy)
        
        # Mercator projection
        rad    = state.settings.pi / 180.       # radians <-> degree
        y_idx  = npx.arange(state.settings.ny + 4) - (state.settings.ny + 4 + 1) / 2 # 4 -> includes ghosts
        
        vs.yt  = update(
            vs.yt,
            at[...],
            1 / rad * npx.arcsin( npx.tanh( state.settings.dxy *rad* y_idx ) )
        )
        vs.dyt  = update(
            vs.dyt,
            at[...],
            npx.cos( rad * vs.yt ) * state.settings.dxy
        )
        vs.dzt = update(
            vs.dzt,
            at[...],
            self.madec_imbard_1996(
                max_depth=4000.,
                hybrid_depth=1000.,
                dzt_surf=10.0,
                inflexion=35.0,
                tanh_slope=10.5,
                nzt=state.settings.nz
            )
        )
        print(vs.dzt)

    @veros_routine
    def set_coriolis(self, state):
        vs = state.variables
        settings = state.settings
        vs.coriolis_t = update(
            vs.coriolis_t, at[...], 2 * settings.omega * npx.sin(vs.yt[None, :] / 180.0 * settings.pi)
        )

    @veros_routine
    def set_topography(self, state):
        vs = state.variables
        x, y = npx.meshgrid(vs.xt, vs.yt, indexing="ij")
        vs.kbot = npx.logical_or(x > 1.0, y < -50).astype("int")

    @veros_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        # initial conditions
        vs.temp = update(vs.temp, at[...], ((1 - vs.zt[None, None, :] / vs.zw[0]) * 15 * vs.maskT)[..., None])
        vs.salt = update(vs.salt, at[...], 35.0 * vs.maskT[..., None])

        # wind stress forcing
        yt_min = global_min(vs.yt.min())
        yu_min = global_min(vs.yu.min())
        yt_max = global_max(vs.yt.max())
        yu_max = global_max(vs.yu.max())

        taux = allocate(state.dimensions, ("yt",))
        taux = npx.where(vs.yt < -20, 0.1 * npx.sin(settings.pi * (vs.yu - yu_min) / (-20.0 - yt_min)), taux)
        taux = npx.where(vs.yt > 10, 0.1 * (1 - npx.cos(2 * settings.pi * (vs.yu - 10.0) / (yu_max - 10.0))), taux)
        vs.surface_taux = taux * vs.maskU[:, :, -1]

        # surface heatflux forcing
        vs.t_star = allocate(state.dimensions, ("yt",), fill=15)
        vs.t_star = npx.where(vs.yt < -20, 15 * (vs.yt - yt_min) / (-20 - yt_min), vs.t_star)
        vs.t_star = npx.where(vs.yt > 20, 15 * (1 - (vs.yt - 20) / (yt_max - 20)), vs.t_star)
        vs.t_rest = vs.dzt[npx.newaxis, -1] / (30.0 * 86400.0) * vs.maskT[:, :, -1]

        if settings.enable_tke:
            vs.forc_tke_surface = update(
                vs.forc_tke_surface,
                at[2:-2, 2:-2],
                npx.sqrt(
                    (0.5 * (vs.surface_taux[2:-2, 2:-2] + vs.surface_taux[1:-3, 2:-2]) / settings.rho_0) ** 2
                    + (0.5 * (vs.surface_tauy[2:-2, 2:-2] + vs.surface_tauy[2:-2, 1:-3]) / settings.rho_0) ** 2
                 )
                ** (1.5),
            )

        if settings.enable_idemix:
            vs.forc_iw_bottom = 1e-6 * vs.maskW[:, :, -1]
            vs.forc_iw_surface = 1e-7 * vs.maskW[:, :, -1]

    @veros_routine
    def set_forcing(self, state):
        vs = state.variables
        vs.forc_temp_surface = vs.t_rest * (vs.t_star - vs.temp[:, :, -1, vs.tau])

    @veros_routine
    def set_diagnostics(self, state):
        settings = state.settings
        diagnostics = state.diagnostics

        diagnostics["snapshot"].output_frequency = 86400 * 10
        diagnostics["averages"].output_variables = (
            "salt",
            "temp",
            "u",
            "v",
            "w",
            "psi",
            "surface_taux",
            "surface_tauy",
        )
        diagnostics["averages"].output_frequency = 365 * 86400.0
        diagnostics["averages"].sampling_frequency = settings.dt_tracer * 10
        diagnostics["overturning"].output_frequency = 365 * 86400.0 / 48.0
        diagnostics["overturning"].sampling_frequency = settings.dt_tracer * 10
        diagnostics["tracer_monitor"].output_frequency = 365 * 86400.0 / 12.0
        diagnostics["energy"].output_frequency = 365 * 86400.0 / 48
        diagnostics["energy"].sampling_frequency = settings.dt_tracer * 10

    @veros_routine
    def after_timestep(self, state):
        pass
    
    @staticmethod
    def mercator_grid(latitude, settings):
        '''
        Compute the number of gridpoints to the equator from a given latitude (MERCATOR).
      
        Find number of gridpoints between the equator and the (approximate) latitude
        for the mercator projection. This is useful to find locations from the physical domain
        on the numerical grid. We can also ensure that the equator is always on a U/T point.
        TODO: clarify if npx.sin is preferred to math.sin.
        '''
        equator = abs(
            180./settings.pi * npx.log(
                npx.cos( settings.pi / 4. - settings.pi / 180. * latitude / 2. ) /
                npx.sin( settings.pi / 4. - settings.pi / 180. * latitude / 2. )
            ) / settings.dxy
        ) * npx.sign(latitude)
        return int(equator)
    
    @staticmethod
    def madec_imbard_1996(max_depth, hybrid_depth, dzt_surf, inflexion, tanh_slope, nzt):
        '''
        Compute the vertical grid spacing as a function of depth.

        Following Madec & Imbard 1996.

        Parameters:
        
        max_depth       (float) : Maximum depth of the domain (default: 4000)
        hybrid_depth    (float) : Connection depth with hybrid vertical coordinates (default: 1000)
            -->  (Not implemented in VEROS, but optional in NEMO and necessary to compute dzt)
        dzt_surf        (float) : Vertical grid spacing at the surface (default: 10.0)
        inflexion       (float) : Inflexion point (default: 35.0)
        tanh_slope      (float) : Slope of the tanh (default: 10.5)
        nzt             (int)   : Number of vertical levels (default: 36)

        Returns:
        dzt (numpy.ndarray) : Vertical grid spacing on T-points.
        TODO: zt, zw, dzw are computed again in numerics.py. Ignored for now, but not optimal.
        '''
        a0 =    (dzt_surf - (max_depth - hybrid_depth) / (nzt - 1)) / \
                (npx.tanh((1 - inflexion) / tanh_slope) - tanh_slope / (nzt -1)) * \
                (npx.log(npx.cosh((nzt - inflexion) / tanh_slope))) * \
                ( - npx.log(npx.cosh((1 - inflexion) / tanh_slope)))

        a1 = dzt_surf - a0 * npx.tanh( ( 1 - inflexion) / tanh_slope )

        a2 = - a1 - a0 * tanh_slope * npx.log( npx.cosh( ( 1 - inflexion ) / tanh_slope ) )

        jk = npx.arange(1, nzt + 1)
        zw = jk.astype(float)
        zt = zw + 0.5

        depth_w = a2 + a1 * zw + a0 * tanh_slope * npx.log(npx.cosh((zw - inflexion) / tanh_slope)) + hybrid_depth
        depth_t = a2 + a1 * zw + a0 * tanh_slope * npx.log(npx.cosh((zt - inflexion) / tanh_slope)) + hybrid_depth
        print(depth_t)
        print(depth_w)
        dzt = npx.diff(depth_t, append=abs(depth_t[-1] - depth_w[-1]))
        return(dzt)