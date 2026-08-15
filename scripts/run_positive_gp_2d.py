"""Positive multi-output GP on a real 500 km March 8 two-dimensional region."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.likelihoods import QuadratureLikelihood
from gpflow.quadrature import NDiagGHQuadrature


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
parser=argparse.ArgumentParser()
parser.add_argument("--data",type=Path,default=ROOT/"data"/"processed"/"real_data_March8_dense_2D_region.csv")
parser.add_argument("--cv-output",type=Path,default=OUT/"positive_gp_2d_blocked_predictions.csv")
parser.add_argument("--grid-output",type=Path,default=OUT/"positive_gp_2d_full_grid.csv")
parser.add_argument("--summary-output",type=Path,default=OUT/"positive_gp_2d_summary.json")
parser.add_argument("--window",default="March_8")
parser.add_argument("--x-mid-km",type=float,default=237.5)
parser.add_argument("--y-mid-km",type=float,default=262.5)
args=parser.parse_args()

logging.getLogger("absl").setLevel(logging.ERROR)
tf.get_logger().setLevel("ERROR")
gpflow.config.set_default_float(np.float64)

rho_w, rho_i, rho_s = 1024.0, 915.0, 300.0
c, c_s = 299792458.0, 229792458.0
H = np.array(
    [
        [(rho_w-rho_i)/rho_w, 1.0-c/c_s-rho_s/rho_w],
        [(rho_w-rho_i)/rho_w, 1.0-rho_s/rho_w],
    ], dtype=np.float64,
)
SIGMA = np.array([0.05, 0.02], dtype=np.float64)
RAW_BASELINE = np.log(np.expm1(np.array([1.8, 0.18], dtype=np.float64)))
LENGTHSCALES_KM = (200.0, 100.0)


class PositivePhysicsLikelihood(QuadratureLikelihood):
    def __init__(self):
        super().__init__(
            input_dim=2,
            latent_dim=2,
            observation_dim=2,
            quadrature=NDiagGHQuadrature(2, 8),
        )
        self.H = tf.convert_to_tensor(H, dtype=tf.float64)
        self.sigma = tf.convert_to_tensor(SIGMA, dtype=tf.float64)

    def _conditional_mean(self, X, F):
        return tf.linalg.matvec(self.H, tf.nn.softplus(F))

    def _conditional_variance(self, X, F):
        return tf.broadcast_to(tf.square(self.sigma), tf.shape(F))

    def _log_prob(self, X, F, Y):
        mean = self._conditional_mean(X, F)
        z = (Y-mean)/self.sigma
        return tf.reduce_sum(
            -0.5*tf.square(z)-tf.math.log(self.sigma)-0.5*np.log(2.0*np.pi), axis=-1
        )


def positive_moments(mu, variance):
    nodes, weights = np.polynomial.hermite.hermgauss(30)
    draws = mu[...,None] + np.sqrt(2.0*np.maximum(variance,0.0))[...,None]*nodes
    values = np.logaddexp(0.0, draws)
    mean = np.sum(values*weights, axis=-1)/np.sqrt(np.pi)
    second = np.sum(values**2*weights, axis=-1)/np.sqrt(np.pi)
    return mean, np.sqrt(np.maximum(second-mean**2,0.0))


def inducing_grid(X):
    gx = np.linspace(X[:,0].min(), X[:,0].max(), 7)
    gy = np.linspace(X[:,1].min(), X[:,1].max(), 7)
    xx, yy = np.meshgrid(gx, gy)
    return np.column_stack([xx.ravel(), yy.ravel()])


def fit_predict(X_train, Y_train, X_predict, steps, seed):
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    kernel = gpflow.kernels.SeparateIndependent(
        [
            gpflow.kernels.Matern52(lengthscales=[0.2,0.2], variance=0.25),
            gpflow.kernels.Matern52(lengthscales=[0.1,0.1], variance=0.05),
        ]
    )
    inducing = gpflow.inducing_variables.SharedIndependentInducingVariables(
        gpflow.inducing_variables.InducingPoints(inducing_grid(X))
    )
    model = gpflow.models.SVGP(
        kernel,
        PositivePhysicsLikelihood(),
        inducing,
        mean_function=gpflow.mean_functions.Constant(RAW_BASELINE),
        num_latent_gps=2,
        q_diag=False,
        num_data=len(X_train),
    )
    for latent_kernel in model.kernel.kernels:
        gpflow.utilities.set_trainable(latent_kernel.lengthscales, False)
    gpflow.utilities.set_trainable(model.inducing_variable, False)
    gpflow.utilities.set_trainable(model.mean_function, False)
    loss = model.training_loss_closure((X_train,Y_train), compile=True)
    natgrad = gpflow.optimizers.NaturalGradient(gamma=0.04)
    gpflow.utilities.set_trainable(model.q_mu, False)
    gpflow.utilities.set_trainable(model.q_sqrt, False)
    adam = tf.optimizers.Adam(learning_rate=0.01)
    for _ in range(steps):
        natgrad.minimize(loss, var_list=[(model.q_mu,model.q_sqrt)])
        with tf.GradientTape() as tape:
            value=loss()
        gradients=tape.gradient(value,model.trainable_variables)
        adam.apply_gradients(zip(gradients,model.trainable_variables))
    raw_mean,raw_variance=model.predict_f(X_predict)
    latent,latent_std=positive_moments(raw_mean.numpy(),raw_variance.numpy())
    return latent,latent_std,latent@H.T,float(loss())


data=np.genfromtxt(args.data,delimiter=",",names=True,dtype=None,encoding="utf-8")
local_x=np.asarray(data["local_x_km"],float)
local_y=np.asarray(data["local_y_km"],float)
X=np.column_stack([local_x,local_y])/1000.0
Y=np.column_stack([data["freeboard_cs2_m"],data["freeboard_is2_m"]]).astype(float)
n=len(X)

x_mid=args.x_mid_km; y_mid=args.y_mid_km
fold_id=(local_x>x_mid).astype(int)+2*(local_y>y_mid).astype(int)
folds=[np.where(fold_id==fold)[0] for fold in range(4)]

point_latent=np.full((n,2),np.nan); point_observation=np.full((n,2),np.nan)
positive_latent=np.full((n,2),np.nan); positive_std=np.full((n,2),np.nan)
positive_observation=np.full((n,2),np.nan); fold_losses=[]

for fold,test_idx in enumerate(folds):
    train_idx=np.setdiff1d(np.arange(n),test_idx)
    train_point=np.linalg.solve(H,Y[train_idx].T).T
    distances=np.sqrt(np.sum((X[test_idx,None,:]-X[train_idx][None,:,:])**2,axis=2))
    nearest=np.argmin(distances,axis=1)
    point_latent[test_idx]=train_point[nearest]
    point_observation[test_idx]=point_latent[test_idx]@H.T
    positive_latent[test_idx],positive_std[test_idx],positive_observation[test_idx],loss=fit_predict(
        X[train_idx],Y[train_idx],X[test_idx],steps=220,seed=20260808+fold
    )
    fold_losses.append({"fold":fold,"test_rows":len(test_idx),"final_loss":loss})
    print(json.dumps(fold_losses[-1]),flush=True)


def metrics(actual,predicted):
    error=predicted-actual
    return {
        "RMSE_m":float(np.sqrt(np.mean(error**2))),
        "MAE_m":float(np.mean(np.abs(error))),
        "bias_m":float(np.mean(error)),
    }


cv_rows=[]
for i in range(n):
    cv_rows.append({
        "row_id":int(data["row_id"][i]),"fold":int(fold_id[i]),
        "x_m":int(data["x_m"][i]),"y_m":int(data["y_m"][i]),
        "local_x_km":local_x[i],"local_y_km":local_y[i],
        "actual_cs2_m":Y[i,0],"actual_is2_m":Y[i,1],
        "point_Hi_m":point_latent[i,0],"point_Hs_m":point_latent[i,1],
        "point_pred_cs2_m":point_observation[i,0],"point_pred_is2_m":point_observation[i,1],
        "positive_gp_Hi_m":positive_latent[i,0],"positive_gp_Hs_m":positive_latent[i,1],
        "positive_gp_Hi_std_m":positive_std[i,0],"positive_gp_Hs_std_m":positive_std[i,1],
        "positive_gp_pred_cs2_m":positive_observation[i,0],
        "positive_gp_pred_is2_m":positive_observation[i,1],
    })
args.cv_output.parent.mkdir(parents=True,exist_ok=True)
with args.cv_output.open("w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=list(cv_rows[0]));writer.writeheader();writer.writerows(cv_rows)

# Full-data model for a visualization grid; this fit is not used for CV metrics.
gx_km=np.linspace(local_x.min(),local_x.max(),45)
gy_km=np.linspace(local_y.min(),local_y.max(),45)
gxx,gyy=np.meshgrid(gx_km,gy_km)
X_grid=np.column_stack([gxx.ravel(),gyy.ravel()])/1000.0
grid_latent,grid_std,grid_observation,full_loss=fit_predict(X,Y,X_grid,steps=280,seed=20260888)
grid_rows=[]
for i in range(len(X_grid)):
    grid_rows.append({
        "local_x_km":X_grid[i,0]*1000,"local_y_km":X_grid[i,1]*1000,
        "positive_gp_Hi_m":grid_latent[i,0],"positive_gp_Hs_m":grid_latent[i,1],
        "positive_gp_Hi_std_m":grid_std[i,0],"positive_gp_Hs_std_m":grid_std[i,1],
        "positive_gp_pred_cs2_m":grid_observation[i,0],"positive_gp_pred_is2_m":grid_observation[i,1],
    })
args.grid_output.parent.mkdir(parents=True,exist_ok=True)
with args.grid_output.open("w",newline="",encoding="utf-8") as handle:
    writer=csv.DictWriter(handle,fieldnames=list(grid_rows[0]));writer.writeheader();writer.writerows(grid_rows)

summary={
    "region":{"window":args.window,"rows":n,"extent_km":[float(local_x.max()-local_x.min()),float(local_y.max()-local_y.min())],
              "occupied_fraction_of_20x20_grid":float(n/400),"quadrant_fold_sizes":[int(len(f)) for f in folds]},
    "fixed_fold_boundaries_km":{"x_mid":x_mid,"y_mid":y_mid},
    "fixed_model":{"sigma_m":{"CS2":SIGMA[0],"IS2":SIGMA[1]},
                   "lengthscales_km":{"Hi":LENGTHSCALES_KM[0],"Hs":LENGTHSCALES_KM[1]},
                   "inducing_points":49},
    "point_nearest":{"CS2":metrics(Y[:,0],point_observation[:,0]),"IS2":metrics(Y[:,1],point_observation[:,1])},
    "positive_gp":{"CS2":metrics(Y[:,0],positive_observation[:,0]),"IS2":metrics(Y[:,1],positive_observation[:,1]),
                   "negative_Hi":int((positive_latent[:,0]<0).sum()),"negative_Hs":int((positive_latent[:,1]<0).sum())},
    "fold_losses":fold_losses,"full_visualization_fit_loss":full_loss,
}
args.summary_output.parent.mkdir(parents=True,exist_ok=True)
args.summary_output.write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
print(args.cv_output);print(args.grid_output);print(args.summary_output)
