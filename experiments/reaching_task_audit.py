"""Exact objective attribution and instantaneous feasibility before hold drift."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.optimize import linprog

from common import ROOT,load_configs,make_model,prepare_paired_initial_condition,write_csv
from reaching_contact_mode import ContactModeController
from se3_whole_body_control.control.whole_body_qp import WholeBodyQPController

LABELS=('torso','pelvis','com','posture','reach','acceleration','torque','old_body_slack','nominal_torque')


class TaskAuditController(ContactModeController):
    def _add_objective(self,P,q,A,b,weight):
        self.objectives.append((A.copy(),b.copy(),float(weight)))
        return super()._add_objective(P,q,A,b,weight)

    def _build_problem(self):
        self.objectives=[]
        problem=super()._build_problem()
        assert len(self.objectives)==len(LABELS), 'objective layout changed; audit labels must be reviewed'
        self.problem=problem
        return problem

    def _constraint_budget_ratio(self,A,x,lower,upper,eps_abs,eps_rel):
        self.solution=np.array(x).copy()
        return WholeBodyQPController._constraint_budget_ratio(A,x,lower,upper,eps_abs,eps_rel)


def objective_records(controller):
    x=controller.solution
    records=[]
    constant=0.0
    for name,(A,b,weight) in zip(LABELS,controller.objectives):
        residual=A@x[:A.shape[1]]-b
        cost=float(weight*(residual@residual))
        constant+=float(weight*(b@b))
        records.append(dict(task=name,weight=weight,residual_norm=float(np.linalg.norm(residual)),
                            residual=residual.tolist(),weighted_squared_residual=cost))
    slack=len(x)-sum(s['J'].shape[0] for s in controller.specs)
    for foot,spec in enumerate(controller.specs):
        count=spec['J'].shape[0]
        values=x[slack:slack+count]
        weight=controller.cfg['qp_slack_weight']/max(1,len(spec['points']))
        records.append(dict(task=f'point_slack_{foot}',weight=weight,residual_norm=float(np.linalg.norm(values)),
                            residual=values.tolist(),weighted_squared_residual=float(weight*(values@values))))
        slack+=count
    regularization=float(0.5e-9*(x@x))
    cost=sum(r['weighted_squared_residual'] for r in records)+regularization
    P,q=controller.problem[:2]
    direct=float(0.5*x@P@x+q@x+constant)
    np.testing.assert_allclose(cost,direct,rtol=1e-9,atol=1e-7)
    for record in records:
        record['cost_share']=record['weighted_squared_residual']/max(cost,1e-30)
    return records,cost,regularization


def constraint_labels(controller):
    nv,nu=controller.model.nv,controller.model.nu
    labels=['dynamics']*nv+['acceleration_bound']*nv+['torque_bound']*nu
    for foot in range(2):
        labels += [f'normal_{foot}']+[f'world_friction_{foot}']*4
    labels += ['old_slack_fixed']*controller.nslack
    for foot,spec in enumerate(controller.specs):
        labels += [f'wrench_synthesis_{foot}']*6
        labels += [f'ray_nonnegative_{foot}']*spec['G'].shape[1]
        labels += [f'point_acceleration_{foot}']*spec['J'].shape[0]
    assert len(labels)==controller.problem[2].shape[0]
    return labels


def active_constraints(controller):
    A,low,high=controller.problem[2:5]
    values=np.asarray(A@controller.solution)
    settings=controller.cfg['solver']
    absolute,relative=settings['eps_abs'],settings['eps_rel']
    records=[]
    for i,(name,value,l,u) in enumerate(zip(constraint_labels(controller),values,low,high)):
        equality=bool(np.isfinite(l) and np.isfinite(u) and l==u)
        lower_active=bool(not equality and np.isfinite(l) and abs(value-l)<=absolute+relative*max(abs(value),abs(l)))
        upper_active=bool(not equality and np.isfinite(u) and abs(value-u)<=absolute+relative*max(abs(value),abs(u)))
        records.append(dict(row=i,group=name,value=float(value),lower=float(l) if np.isfinite(l) else None,
                            upper=float(u) if np.isfinite(u) else None,equality=equality,
                            lower_active=lower_active,upper_active=upper_active))
    return records


def com_feasibility(controller,preserve):
    """LP diagnostic, not a new policy: keep the original point slack fixed."""
    x=controller.solution
    A,low,high=controller.problem[2:5]
    A=A.toarray()
    equal=np.isfinite(low)&np.isfinite(high)&(low==high)
    upper=np.isfinite(high)&~equal
    lower=np.isfinite(low)&~equal
    eq=A[equal].tolist(); rhs=low[equal].tolist()
    slack=len(x)-sum(s['J'].shape[0] for s in controller.specs)
    for i in range(slack,len(x)):
        row=np.zeros(len(x)); row[i]=1
        eq.append(row); rhs.append(float(x[i]))
    objectives=dict(zip(LABELS,controller.objectives))
    for name in preserve:
        task,_,_=objectives[name]
        padded=np.pad(task,((0,0),(0,len(x)-task.shape[1])))
        eq.extend(padded); rhs.extend(padded@x)
    com,target,_=objectives['com']
    com=np.pad(com,((0,0),(0,len(x)-com.shape[1])))
    inequalities=np.vstack([A[upper],-A[lower]])
    bounds=np.r_[high[upper],-low[lower]]
    # Three auxiliary errors minimize L1 task deviation, without changing a QP gain.
    inequalities=np.vstack([np.pad(inequalities,((0,0),(0,3))),np.c_[com,-np.eye(3)],np.c_[-com,-np.eye(3)]])
    bounds=np.r_[bounds,target,-target]
    result=linprog(np.r_[np.zeros(len(x)),np.ones(3)],A_ub=inequalities,b_ub=bounds,
                   A_eq=np.pad(np.array(eq),((0,0),(0,3))),b_eq=np.array(rhs),
                   bounds=[(None,None)]*len(x)+[(0,None)]*3,method='highs')
    record=dict(preserved_attained_tasks=list(preserve),success=bool(result.success),status=result.message)
    if result.success:
        candidate=result.x[:len(x)]
        residual=com@candidate-target
        ratio=WholeBodyQPController._constraint_budget_ratio(controller.problem[2],candidate,low,high,
                                                            controller.cfg['solver']['eps_abs'],controller.cfg['solver']['eps_rel'])
        record.update(minimum_com_L1_residual=float(np.linalg.norm(residual,ord=1)),
                      com_residual_norm=float(np.linalg.norm(residual)),constraint_budget_ratio=ratio,
                      torque_peak_Nm=float(np.max(abs(candidate[controller.model.nv:controller.model.nv+controller.model.nu]))),
                      max_qdd=float(np.max(abs(candidate[:controller.model.nv]))))
        assert ratio<=1
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=load_configs(ROOT,robot_name='unitree_g1')
    initial=prepare_paired_initial_condition(cfg)
    path=ROOT/'results/reaching_contact/mode_pilot/trials/front_12.5cm_contact_mode_hold'
    with np.load(path/'trajectory.npz',allow_pickle=False) as payload:
        arrays={key:payload[key] for key in payload.files}
    summary=json.loads((path/'summary.json').read_text())
    records,flat=[],[]
    for time in (2.996,3.020,3.080,3.152,3.240,3.320,3.400,3.436):
        i=int(np.argmin(abs(arrays['time_s']-time)))
        model=make_model(cfg)
        model.reset(arrays['qpos_history'][0],arrays['qvel_history'][0])
        controller=TaskAuditController(model,cfg['controller'],summary['provenance']['reaching'])
        controller.q_des=arrays['joint_reference_history'][i].copy()
        controller.pd_fallback.q_des=controller.q_des.copy()
        controller.T_des_torso,controller.T_des_pelvis=initial.desired_torso.copy(),initial.desired_pelvis.copy()
        controller.com_des=initial.com_reference.copy()
        model.reset(arrays['qpos_history'][i],arrays['qvel_history'][i])
        model.data.time=float(arrays['time_s'][i])
        result=controller.solve()
        assert result.success,result.message
        np.testing.assert_allclose(result.control,arrays['control'][i],rtol=1e-6,atol=1e-6)
        objectives,cost,regularization=objective_records(controller)
        constraints=active_constraints(controller)
        diagnostic=[com_feasibility(controller,keep) for keep in ((),('reach',),('reach','torso','pelvis'))]
        record=dict(time_s=float(arrays['time_s'][i]),com_xy_displacement_m=float(np.linalg.norm(model.center_of_mass()[:2]-controller.com_des[:2])),
                    contact_motion_ranks=[s['rank'] for s in controller.specs],objectives=objectives,
                    objective_sum=cost,regularization=regularization,constraints=constraints,com_feasibility=diagnostic,
                    control_replay_max_error_Nm=float(np.max(abs(result.control-arrays['control'][i]))),
                    constraint_budget_ratio=result.diagnostics['constraint_budget_ratio'])
        records.append(record)
        flat.extend(dict(time_s=record['time_s'],**objective) for objective in objectives)
        print(record['time_s'],record['com_xy_displacement_m'],[(r['task'],round(r['cost_share'],4)) for r in objectives],flush=True)
        print('LP',[(d['preserved_attained_tasks'],d.get('minimum_com_L1_residual'),d['success']) for d in diagnostic],flush=True)
    write_csv(flat,args.output/'objectives.csv')
    report=dict(records=records,physical_trials_run=0,controller_changed=False,task_weights_changed=False,
                source_version=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                trajectory_path=path.relative_to(ROOT).as_posix(),trajectory_sha256=hashlib.sha256((path/'trajectory.npz').read_bytes()).hexdigest(),
                active_definition='Inequality distance to bound within original solver absolute plus relative tolerance; equality rows separate; zero ray coefficients do not alone establish a binding physical support limit',
                limitation='Weighted cost share is not causal importance. LP is instantaneous, fixes point slack, preserves attained task accelerations rather than task targets, and can choose extreme joint accelerations. No rollout or gain change.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.output.iterdir() if p.is_file()]
    (args.output/'manifest.json').write_text(json.dumps(dict(files=files),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
