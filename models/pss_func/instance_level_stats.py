def calc_acc(preds, gold_field, pred_field, top_k=1, divergent=None, role_correct=None, func_correct=None):
    assert gold_field in ['role', 'func']
    assert pred_field in ['role', 'func']

    fpreds = preds
    if divergent is not None:
        fpreds = [p for p in fpreds if (p.gold_role != p.gold_func) == divergent]
    if role_correct is not None:
        fpreds = [p for p in fpreds if (p.gold_role == p.pred_role) == role_correct]
    if func_correct is not None:
        fpreds = [p for p in fpreds if (p.gold_func == p.pred_func) == func_correct]

    rest = [p for p in preds if p not in fpreds]

    correct = 0
    for pred in fpreds:
        pred_dist = getattr(pred, 'pred_' + pred_field + '_dist')
        top_k_pred = sorted(pred_dist.keys(), key=lambda x: -pred_dist[x])[:top_k]
        if getattr(pred, 'gold_' + gold_field) in top_k_pred:
            correct += 1

    if len(fpreds) == len(preds):
        total_correct = correct
    else:
        total_correct = calc_acc(preds, gold_field, pred_field, top_k)[0] * len(preds)

    errors = len(fpreds) - correct
    total_errors = len(preds) - total_correct

    return correct / len(fpreds) if len(fpreds) else -1, errors / total_errors if total_errors else -1, len(fpreds)

def print_report(preds):
    print('Role Acc: %1.2f' % calc_acc(preds, 'role', 'role')[0])
    print('Func Acc (using role for prediction): %1.2f' % calc_acc(preds, 'func', 'role')[0])
    for top_k in [2,3,4]:
        print('Func Acc (using role for prediction, top_k=%d): %1.2f' % (top_k, calc_acc(preds, 'func', 'role', top_k=top_k)[0]))
    print('')
    print('Func Acc. Breakdown:')
    print('--------------------')
    print('%s\t%s\t%s\t%s\t%s' % ('Divergent', 'Role Correct', '% of Samples', 'Acc.', '% of Error'))
    for divergent in [True, False]:
        for role_correct in [True, False]:
            acc, err, cnt = calc_acc(preds, 'func', 'role', divergent=divergent, role_correct=role_correct)
            print('%s\t%s\t%2.2f\t%1.2f\t%2.2f' % (divergent, role_correct, cnt / len(preds) * 100, acc, err * 100))
    print('')
    print('Func Acc. Breakdown (top_k=2):')
    print('-----------------------------')
    print('%s\t%s\t%s\t%s\t%s' % ('Divergent', 'Role Correct', '% of Samples', 'Acc.', '% of Error'))
    for divergent in [True, False]:
        for role_correct in [True, False]:
            acc, err, cnt = calc_acc(preds, 'func', 'role', divergent=divergent, role_correct=role_correct, top_k=2)
            print('%s\t%s\t%2.2f\t%1.2f\t%2.2f' % (divergent, role_correct, cnt / len(preds) * 100, acc, err * 100))

    # print('Func Acc. Breakdown:')
    # print('----------')
    # print('%s\t%s\t%s\t%s\t%s' % ('Divergent', 'Role Correct', 'Func Correct', 'Acc.', 'Error Percentage'))
    # for divergent in [True, False]:
    #     for role_correct in [True, False]:
    #         for func_correct in [True, False]:
    #             acc, err = calc_acc(preds, 'func', 'role', divergent=divergent, role_correct=role_correct, func_correct=func_correct)
    #             print('%s\t%s\t%s\t%1.2f\t%2.2f' % (divergent, role_correct, func_correct, acc, err * 100))