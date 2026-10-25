"""CPU-only aggregate diagnostics. This module performs no model forward or I/O."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

import torch

LOSSES = ('classification', 'cc', 'mscl', 'cscl')
AUXILIARIES = ('cc', 'mscl', 'cscl')
RELATIONS = ('t', 'a', 'v', 'intra', 'inter')
BRANCHES = ('both_wrong_default0', 'wrong_to_correct_explicit0', 'correct_to_wrong_explicit1',
            'both_correct_removed_conf_ge_main_explicit0', 'both_correct_removed_conf_lower_explicit1')


def _context(context):
    """Public context is scalar metadata, never dialogue IDs or utterance contents."""
    result = dict(context)
    forbidden = ('dialogue', 'utterance', 'speaker_id', 'video_id', 'sample_id', 'example_id', 'text', 'audio_path')
    for key, value in result.items():
        if not isinstance(key, str) or any(word in key.lower() for word in forbidden):
            raise ValueError('Sensitive or unsupported public context key: ' + str(key))
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError('Public context values must be scalar metadata')
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite context metadata')
    return result


def _tensor(value):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
        raise ValueError('Diagnostic inputs must be CPU tensors')
    return value.detach()


def _finite(value):
    return bool(torch.isfinite(value).all().item())


def _norm(value):
    return float(value.double().square().sum().sqrt().item())


def parameter_groups(parameter_names):
    """Group model.named_parameters() names; buffers must not be passed as parameters."""
    names = list(parameter_names)
    if len(names) != len(set(names)) or any(not isinstance(name, str) for name in names):
        raise ValueError('Parameter names must be unique strings')
    if any(name.endswith(('running_mean', 'running_var', 'num_batches_tracked')) for name in names):
        raise ValueError('BatchNorm buffers are not parameters')
    groups = {name: [] for name in ('text_gru', 'audio_projection', 'visual_projection', 'graph_shared', 'classification_head', 'other')}
    for name in names:
        target = ('text_gru' if name.startswith('gru_l.') else
                  'audio_projection' if name.startswith('linear_a.') else
                  'visual_projection' if name.startswith('linear_v.') else
                  'graph_shared' if name.startswith('graph_model.') else
                  'classification_head' if name.startswith(('smax_fc.', 'hidfc.')) else 'other')
        groups[target].append(name)
    heads = set(groups['classification_head'])
    groups['shared_backbone'] = [name for name in names if name not in heads]
    groups['all'] = names
    return groups


def _coefficient(coefficients, loss):
    value = float(coefficients.get(loss, 1. if loss == 'classification' else 0.))
    if not math.isfinite(value) or value < 0:
        raise ValueError('Coefficients must be finite and nonnegative')
    return value


def _gradient_inputs(gradients, groups):
    if set(gradients) != set(LOSSES):
        raise ValueError('Gradients require classification/cc/mscl/cscl maps')
    names = set(name for values in groups.values() for name in values)
    for loss in LOSSES:
        if not isinstance(gradients[loss], Mapping):
            raise ValueError('Each gradient loss must map parameter names to tensors or None')
        for value in gradients[loss].values():
            if value is not None:
                _tensor(value)
    for name in names:
        shapes = {tuple(gradients[loss][name].shape) for loss in LOSSES if gradients[loss].get(name) is not None}
        if len(shapes) > 1:
            raise ValueError('Gradient shape mismatch for parameter ' + name)


def _auxiliary_sums(gradients, groups, coefficients):
    result, support = {}, {}
    for name in dict.fromkeys(name for values in groups.values() for name in values):
        values = []
        record = {'source_missing': 0, 'inactive_source_tensors': 0, 'active_source_tensors': 0}
        for loss in AUXILIARIES:
            value, coefficient = gradients[loss].get(name), _coefficient(coefficients, loss)
            record['source_missing'] += value is None
            if value is None:
                continue
            if coefficient == 0:
                record['inactive_source_tensors'] += 1
                continue
            record['active_source_tensors'] += 1
            values.append(_tensor(value).double() * coefficient)
        # None means no active auxiliary path, rather than a fabricated zero gradient.
        result[name] = sum(values[1:], values[0].clone()) if values else None
        support[name] = record
    return result, support


def _norm_record(values, names):
    present = [values.get(name) for name in names if values.get(name) is not None]
    missing = len(names) - len(present)
    nonfinite = sum(not _finite(value) for value in present)
    zeros = sum(_finite(value) and _norm(value) == 0 for value in present)
    bad_coordinates = sum(int((~torch.isfinite(value)).sum().item()) for value in present)
    raw = None if not present or nonfinite else math.sqrt(sum(_norm(value)**2 for value in present))
    derived_nonfinite = raw is not None and not math.isfinite(raw)
    if derived_nonfinite:
        raw = None
    status = ('empty_group' if not names else 'missing' if not present else 'nonfinite' if nonfinite else 'nonfinite_aggregate' if derived_nonfinite else
              'zero' if raw == 0 else 'partial_missing' if missing else 'ok')
    return {'status': status, 'raw_l2': raw, 'total_tensors': len(names), 'present_tensors': len(present),
            'missing_tensors': missing, 'zero_tensors': zeros, 'nonfinite_tensors': nonfinite,
            'derived_nonfinite': derived_nonfinite,
            'nonfinite_coordinates': bad_coordinates, 'present_coordinates': sum(value.numel() for value in present)}


def gradient_norm_rows(gradients, groups, context, reference_coefficients, actual_coefficients):
    """Raw norms and explicitly distinct offline-reference versus active-objective scales."""
    context = _context(context)
    _gradient_inputs(gradients, groups)
    reference_sum, reference_support = _auxiliary_sums(gradients, groups, reference_coefficients)
    actual_sum, actual_support = _auxiliary_sums(gradients, groups, actual_coefficients)
    maps = {**gradients, 'aux_reference_sum': reference_sum, 'aux_actual_sum': actual_sum}
    rows = []
    for loss, values in maps.items():
        summed = loss.startswith('aux_')
        reference = 1. if summed else _coefficient(reference_coefficients, loss)
        actual = 1. if summed else _coefficient(actual_coefficients, loss)
        for group, names in groups.items():
            record = _norm_record(values, names)
            raw = record['raw_l2']
            support = reference_support if loss == 'aux_reference_sum' else actual_support if loss == 'aux_actual_sum' else None
            rows.append({**context, 'loss': loss, 'group': group, **record,
                'reference_coefficient': reference, 'actual_coefficient': actual,
                'reference_weighted_l2': None if raw is None else reference*raw,
                'actual_weighted_l2': None if raw is None else actual*raw,
                'actual_objective_active': actual != 0,
                'vector_semantics': 'offline_reference_hypothesis' if loss=='aux_reference_sum' else 'actual_auxiliary_sum' if loss=='aux_actual_sum' else 'raw_loss_gradient',
                'source_missing_tensors': None if support is None else sum(support[name]['source_missing'] for name in names),
                'inactive_source_tensors': None if support is None else sum(support[name]['inactive_source_tensors'] for name in names),
                'active_source_tensors': None if support is None else sum(support[name]['active_source_tensors'] for name in names)})
    return rows


def gradient_pair_rows(gradients, groups, context, reference_coefficients, actual_coefficients):
    """Compare whole finite nonzero parameter tensors on shared support, never selected coordinates."""
    context = _context(context)
    _gradient_inputs(gradients, groups)
    reference_sum, _ = _auxiliary_sums(gradients, groups, reference_coefficients)
    actual_sum, _ = _auxiliary_sums(gradients, groups, actual_coefficients)
    maps = {**{loss: gradients[loss] for loss in AUXILIARIES},
            'aux_reference_sum': reference_sum, 'aux_actual_sum': actual_sum}
    rows = []
    for loss, values in maps.items():
        reference = 1. if loss.startswith('aux_') else _coefficient(reference_coefficients, loss)
        actual = 1. if loss.startswith('aux_') else _coefficient(actual_coefficients, loss)
        for group, names in groups.items():
            missing = zeros = bad = tensors = coordinates = 0
            left_missing = right_missing = left_bad = right_bad = 0
            dot = square_a = square_b = 0.
            for name in names:
                left, right = gradients['classification'].get(name), values.get(name)
                left_missing += left is None
                right_missing += right is None
                left_bad += left is not None and not _finite(left)
                right_bad += right is not None and not _finite(right)
                if left is None or right is None:
                    missing += 1
                    continue
                if not _finite(left) or not _finite(right):
                    bad += 1
                    continue
                a, b = left.double().reshape(-1), right.double().reshape(-1)
                if _norm(a) == 0 or _norm(b) == 0:
                    zeros += 1
                    continue
                tensors += 1
                coordinates += a.numel()
                dot += float(torch.dot(a,b).item())
                square_a += float(torch.dot(a,a).item())
                square_b += float(torch.dot(b,b).item())
            norm_a, norm_b = math.sqrt(square_a), math.sqrt(square_b)
            numeric = bool(tensors) and all(math.isfinite(value) for value in (dot,norm_a,norm_b))
            cosine = max(-1.,min(1.,dot/(norm_a*norm_b))) if numeric else None
            status = ('nonfinite_aggregate' if tensors and not numeric else 'partial_nonfinite' if tensors and bad else 'partial_support' if tensors and (missing or zeros) else 'ok' if tensors else
                      'no_common_nonfinite' if bad else 'no_common_missing' if missing else 'no_common_zero' if zeros else 'empty_group')
            rows.append({**context, 'loss_a': 'classification', 'loss_b': loss, 'group': group, 'status': status,
                'common_tensors': tensors, 'common_coordinates': coordinates,
                'excluded_missing_tensors': missing, 'excluded_zero_tensors': zeros, 'excluded_nonfinite_tensors': bad,
                'left_missing_tensors':left_missing,'right_missing_tensors':right_missing,
                'left_nonfinite_tensors':left_bad,'right_nonfinite_tensors':right_bad,
                'exclusion_precedence':'missing, then nonfinite, then zero; side counts retain overlapping causes',
                'norm_a_common': norm_a if numeric else None, 'norm_b_common': norm_b if numeric else None,
                'dot_raw': dot if numeric else None, 'cosine': cosine,
                'negative_cosine': None if cosine is None else cosine < 0,
                'reference_coefficient': reference, 'actual_coefficient': actual,
                'dot_reference_weighted': reference*dot if numeric else None,
                'dot_actual_weighted': actual*dot if numeric else None,
                'cosine_reference_weighted': cosine if reference and numeric else None,
                'cosine_actual_weighted': cosine if actual and numeric else None,
                'norm_ratio_reference_weighted': reference*norm_b/norm_a if numeric else None,
                'norm_ratio_actual_weighted': actual*norm_b/norm_a if numeric else None,
                'vector_semantics': 'offline_reference_hypothesis' if loss=='aux_reference_sum' else 'actual_auxiliary_sum' if loss=='aux_actual_sum' else 'raw_loss_gradient'})
    return rows


def _summary(values):
    if not values:
        return {'mean': None, 'std': None, 'min': None, 'max': None}
    value = torch.tensor(values,dtype=torch.float64)
    return {'mean':float(value.mean()),'std':float(value.std(unbiased=False)),
            'min':float(value.min()),'max':float(value.max())}


def pseudo_rows(context, labels, main_log_probs, removed_log_probs, pseudo_labels, base_logits, class_weights):
    """Audit original binary relation labels and contrastive aggregates without changing their formula."""
    context = _context(context)
    raw_labels, main = _tensor(labels).reshape(-1), _tensor(main_log_probs)
    if not _finite(raw_labels) or not torch.equal(raw_labels,raw_labels.long().to(raw_labels.dtype)):
        raise ValueError('Emotion labels must be finite integer class indices')
    labels = raw_labels.long()
    n = labels.numel()
    if n < 1 or tuple(main.shape)!=(n,7) or not _finite(main) or bool((main>1e-5).any()) or bool(((labels<0)|(labels>6)).any()):
        raise ValueError('Expected nonempty emotion labels and finite [N,7] log probabilities')
    if any(set(value)!=set(RELATIONS) for value in (removed_log_probs,pseudo_labels,base_logits,class_weights)):
        raise ValueError('All relation maps must have t/a/v/intra/inter keys')
    output = {name: [] for name in ('pseudo_counts','pair_counts','class_relations','branch_counts',
                                    'confidence_summary','contribution_summary','private_rows')}
    main_pred, index = main.argmax(-1), torch.arange(n)
    main_gt = main[index,labels].double().exp()
    weighted_original = {}
    for relation in RELATIONS:
        removed = _tensor(removed_log_probs[relation])
        raw_pseudo = _tensor(pseudo_labels[relation]).reshape(-1)
        if not _finite(raw_pseudo) or not torch.equal(raw_pseudo,raw_pseudo.long().to(raw_pseudo.dtype)):
            raise ValueError('Relation labels must be finite integers')
        pseudo = raw_pseudo.long()
        base = _tensor(base_logits[relation]).reshape(-1)
        if tuple(removed.shape)!=(n,7) or not _finite(removed) or bool((removed>1e-5).any()) or pseudo.numel()!=n or base.numel()!=n:
            raise ValueError('Relation tensor shape or finite log-probability mismatch')
        if bool(((pseudo!=0)&(pseudo!=1)).any()):
            raise ValueError('Pseudo labels must be binary relation labels')
        removed_pred = removed.argmax(-1)
        removed_gt = removed[index,labels].double().exp()
        delta = removed_gt-main_gt
        correct, removed_correct = main_pred==labels, removed_pred==labels
        branches, expected = [], []
        for i in range(n):
            if not correct[i] and not removed_correct[i]:
                branch, target = BRANCHES[0],0
            elif not correct[i] and removed_correct[i]:
                branch,target = BRANCHES[1],0
            elif correct[i] and not removed_correct[i]:
                branch,target = BRANCHES[2],1
            elif removed[i,labels[i]] >= main[i,labels[i]]:
                branch,target = BRANCHES[3],0
            else:
                branch,target = BRANCHES[4],1
            branches.append(branch)
            expected.append(target)
        if pseudo.tolist()!=expected:
            raise ValueError('Supplied pseudo labels disagree with the exact original correctness/confidence branches')
        counts = Counter(expected)
        supplied_weights = [float(value) for value in class_weights[relation]]
        if not 1<=len(supplied_weights)<=2 or any(value!=0. for label,value in enumerate(supplied_weights) if label not in counts):
            raise ValueError('Absent binary relation classes must have zero weight when represented')
        for label,count in counts.items():
            if label>=len(supplied_weights) or not math.isclose(supplied_weights[label],n/count,rel_tol=1e-6,abs_tol=1e-6):
                raise ValueError('Class weights must equal original N / relation-class count')
        if any(not math.isfinite(value) for value in supplied_weights):
            raise ValueError('Nonfinite relation class weight')
        weight = torch.tensor([supplied_weights[label] for label in expected],dtype=torch.get_default_dtype())
        weighted_original[relation] = weight*base
        contribution = -weight.double()*base.double()/n
        common = {**context,'relation':relation,'relation_loss':'mscl' if relation in ('t','a','v') else 'cscl',
                  'pseudo_semantics':'binary_correctness_confidence_relation'}
        def select(**requirements):
            return [i for i in range(n) if all((expected[i] if key=='pseudo' else int(labels[i]) if key=='emotion' else branches[i])==value for key,value in requirements.items())]
        def signs(indices):
            return {'n_delta_positive':sum(delta[i]>0 for i in indices).item() if indices else 0,
                    'n_delta_zero':sum(delta[i]==0 for i in indices).item() if indices else 0,
                    'n_delta_negative':sum(delta[i]<0 for i in indices).item() if indices else 0}
        for pseudo_label in (0,1):
            chosen = select(pseudo=pseudo_label)
            default = [i for i in chosen if branches[i]==BRANCHES[0]]
            sign = signs(default)
            output['pseudo_counts'].append({**common,'pseudo_label':pseudo_label,'n_utterances':n,'n_pseudo':len(chosen),
                'fraction':len(chosen)/n,'class_weight':supplied_weights[pseudo_label] if pseudo_label<len(supplied_weights) else 0.,
                'positive_pairs_per_anchor':len(chosen)-1 if chosen else None,
                'negative_pairs_per_anchor':n-len(chosen) if chosen else None,
                'n_default_zero':len(default),'n_main_correct':sum(bool(correct[i]) for i in chosen),
                'n_main_wrong':sum(not bool(correct[i]) for i in chosen),
                **{'default_zero_'+key.removeprefix('n_'):value for key,value in sign.items()}})
        pos_counts = [counts[label]-1 for label in expected]
        neg_counts = [n-counts[label] for label in expected]
        positive = sum(pos_counts)
        negative = sum(neg_counts)
        same_mask = torch.tensor(1.,dtype=torch.float32)+1e-7
        different_mask = torch.tensor(0.,dtype=torch.float32)+1e-7
        output['pair_counts'].append({**common,'n_anchors':n,'ordered_offdiag_pairs':n*(n-1),
            'true_positive_pairs':positive,'true_negative_pairs':negative,
            'zero_positive_anchors':sum(value==0 for value in pos_counts),'zero_negative_anchors':sum(value==0 for value in neg_counts),
            'zero_positive_fraction':sum(value==0 for value in pos_counts)/n,'zero_negative_fraction':sum(value==0 for value in neg_counts)/n,
            'pos_per_anchor_min':min(pos_counts),'pos_per_anchor_max':max(pos_counts),'pos_per_anchor_mean':positive/n,
            'neg_per_anchor_min':min(neg_counts),'neg_per_anchor_max':max(neg_counts),'neg_per_anchor_mean':negative/n,
            'same_class_negative_weight_cells':positive,'different_class_positive_epsilon_cells':negative,
            'same_positive_mask_weight':float(same_mask),'same_negative_mask_weight':float(1-same_mask),
            'different_positive_mask_weight':float(different_mask),'different_negative_mask_weight':float(1-different_mask),
            'pair_count_semantics':'ordered off-diagonal pairs before epsilon; no emotion-consistency interpretation'})
        for emotion in range(7):
            for pseudo_label in (0,1):
                chosen=select(emotion=emotion,pseudo=pseudo_label)
                output['class_relations'].append({**common,'emotion_class':emotion,'pseudo_label':pseudo_label,
                    'n_utterances':len(chosen),'n_main_correct':sum(bool(correct[i]) for i in chosen),
                    'n_removed_correct':sum(bool(removed_correct[i]) for i in chosen),
                    'n_default_zero':sum(branches[i]==BRANCHES[0] for i in chosen)})
        for branch in BRANCHES:
            chosen=select(branch=branch)
            output['branch_counts'].append({**common,'branch':branch,'pseudo_expected':0 if branch in (BRANCHES[0],BRANCHES[1],BRANCHES[3]) else 1,
                'n_utterances':len(chosen),**signs(chosen)})
        subsets={'all':list(range(n)),'default0':select(branch=BRANCHES[0]),
                 'explicit0':[i for i in range(n) if expected[i]==0 and branches[i]!=BRANCHES[0]],
                 'explicit1':[i for i in range(n) if expected[i]==1]}
        for subset,chosen in subsets.items():
            summary=_summary([float(delta[i]) for i in chosen])
            output['confidence_summary'].append({**common,'subset':subset,'n_utterances':len(chosen),
                'mean_main_gt_probability':_summary([float(main_gt[i]) for i in chosen])['mean'],
                'mean_removed_gt_probability':_summary([float(removed_gt[i]) for i in chosen])['mean'],
                **{'gt_probability_delta_'+key:value for key,value in summary.items()},
                'mean_gt_probability_delta':summary['mean'],'std_gt_probability_delta':summary['std'],**signs(chosen)})
        strata=[('all',None,None,None,list(range(n)))]
        strata += [('pseudo',None,label,None,select(pseudo=label)) for label in (0,1)]
        strata += [('emotion',emotion,None,None,select(emotion=emotion)) for emotion in range(7)]
        strata += [('emotion_pseudo',emotion,label,None,select(emotion=emotion,pseudo=label)) for emotion in range(7) for label in (0,1)]
        strata += [('branch',None,None,branch,select(branch=branch)) for branch in BRANCHES]
        for stratum,emotion,pseudo_label,branch,chosen in strata:
            good_base=[i for i in chosen if bool(torch.isfinite(base[i]))]
            good=[i for i in chosen if bool(torch.isfinite(contribution[i]))]
            bad=len(chosen)-len(good)
            finite_sum=sum(float(contribution[i]) for i in good)
            raw_return=-weighted_original[relation].mean()
            fallback=not bool(torch.isfinite(raw_return))
            output['contribution_summary'].append({**common,'stratum':stratum,'emotion_class':emotion,'pseudo_label':pseudo_label,'branch':branch,
                'n_utterances':len(chosen),'n_finite_base_logits':len(good_base),'n_nonfinite_base_logits':len(chosen)-len(good_base),
                'nonfinite_base_logit_fraction':(len(chosen)-len(good_base))/len(chosen) if chosen else None,
                'n_nonfinite_contributions':bad,'contribution_sum':None if bad else finite_sum,'finite_contribution_sum':finite_sum,
                'status':'nonfinite' if bad else 'empty' if not chosen else 'ok','N_denominator':n,
                'raw_loss_before_fallback':None if fallback else float(raw_return),
                'original_nonfinite_fallback_triggered':None,'original_return_after_fallback':None,
                'single_relation_subtotal_nonfinite':fallback,
                'fallback_scope':'offline single-relation subtotal; no original wrapper fallback is applied to this relation alone',
                'contribution_arithmetic':'offline float64 -weight*base_logit/N; original fallback checked in source tensor arithmetic'})
        for i in range(n):
            output['private_rows'].append({**common,'utterance_index':i,'emotion_class':int(labels[i]),
                'pseudo_label':expected[i],'main_pred':int(main_pred[i]),'removed_pred':int(removed_pred[i]),
                'main_gt_probability':float(main_gt[i]),'removed_gt_probability':float(removed_gt[i]),
                'gt_probability_delta':float(delta[i]),'branch':branches[i],'default_zero':branches[i]==BRANCHES[0],
                'true_positive_pairs_for_anchor':pos_counts[i],'true_negative_pairs_for_anchor':neg_counts[i],
                'class_weight':float(weight[i]),'raw_base_logit':float(base[i]) if bool(torch.isfinite(base[i])) else None,
                'raw_base_nonfinite':not bool(torch.isfinite(base[i])),
                'contribution':float(contribution[i]) if bool(torch.isfinite(contribution[i])) else None})
    for name,relations in [('mscl_combined',('t','a','v')),('cscl_combined',('intra','inter'))]:
        combined=sum([weighted_original[relation] for relation in relations][1:],weighted_original[relations[0]].clone())
        raw=-combined.mean()
        fallback=not bool(torch.isfinite(raw))
        good=torch.isfinite(combined)
        finite_sum=float((-combined[good].double()/n).sum())
        output['contribution_summary'].append({**context,'relation':name,'relation_loss':name.split('_')[0],
            'pseudo_semantics':'binary_correctness_confidence_relation','stratum':'all','emotion_class':None,'pseudo_label':None,'branch':None,
            'n_utterances':n,'n_finite_base_logits':None,'n_nonfinite_base_logits':None,
            'n_nonfinite_contributions':int((~good).sum()),'contribution_sum':None if not bool(good.all()) else float((-combined.double()/n).sum()),
            'finite_contribution_sum':finite_sum,'status':'nonfinite' if fallback else 'ok','N_denominator':n,
            'raw_loss_before_fallback':None if fallback else float(raw),
            'original_nonfinite_fallback_triggered':fallback,'original_return_after_fallback':0. if fallback else float(raw),
            'fallback_evidence':'source-arithmetic reconstruction from captured base logits; runner wrapper observations take precedence',
            'fallback_scope':'actual original combined relation loss','contribution_arithmetic':'source dtype weighted sum; offline float64 reporting'})
    # Every table has a uniform CSV schema, including reconstructed combined-loss rows.
    for values in output.values():
        columns = list(dict.fromkeys(key for value in values for key in value))
        for value in values:
            for key in columns:
                value.setdefault(key,None)
    return output
