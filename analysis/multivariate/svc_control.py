# confounders
# con_1: followed by same category
# con_2: followed by the same modality
# con_3: followed by same category and modality
# con_4: followed by the same trial
# con_5: trial number
# con_6: run number
def get_confounder(con):
    sub = [True, True]
    idx1 = []
    idx2 = []
    idx3 = []
    idx4 = []
    idx5 = []
    idx6 = []
    idx7 = []
    idx8 = []

    run_con_3 = np.zeros(shape=events_of_interest.shape)
    run_con_4 = np.zeros(shape=events_of_interest.shape)
    run_con_5 = np.zeros(shape=events_of_interest.shape)
    run_con_6 = np.zeros(shape=events_of_interest.shape)
    run_con_7 = np.zeros(shape=events_of_interest.shape)
    run_con_8 = np.zeros(shape=events_of_interest.shape)

    a, b, c, d, e, f, g, h = [1, 2, 3, 4, 5, 6, 7, 8]

    if con == 'con_1' or con == 'con_2':

        if con == 'con_1':
            a, b, c, d = [1, 2, 5, 6]
            e, f, g, h = [3, 4, 7, 8]

        elif con == 'con_2':
            a, b, c, d = [1, 2, 3, 4]
            e, f, g, h = [4, 5, 7, 8]

        ax1 = np.logical_or(np.logical_or(events_of_interest == a, events_of_interest == b),
                            np.logical_or(events_of_interest == c, events_of_interest == d))
        ax2 = np.logical_or(np.logical_or(events_of_interest == e, events_of_interest == f),
                            np.logical_or(events_of_interest == g, events_of_interest == h))

    elif con == 'con_3' or con == 'con_4':

        if con == 'con_3':

            ax1 = np.logical_or(events_of_interest == a, events_of_interest == b)
            ax2 = np.logical_or(events_of_interest == c, events_of_interest == d)
            ax3 = np.logical_or(events_of_interest == e, events_of_interest == f)
            ax4 = np.logical_or(events_of_interest == g, events_of_interest == h)

        elif con == 'con_4':

            ax1 = events_of_interest == 1
            ax2 = events_of_interest == 2
            ax3 = events_of_interest == 3
            ax4 = events_of_interest == 4
            ax5 = events_of_interest == 5
            ax6 = events_of_interest == 6
            ax7 = events_of_interest == 7
            ax8 = events_of_interest == 8

            [idx5.append(x + 1) for x in range(len(ax5) - 1) if (ax5[x:x + len(sub)] == sub).all()]
            [idx6.append(x + 1) for x in range(len(ax6) - 1) if (ax6[x:x + len(sub)] == sub).all()]
            [idx7.append(x + 1) for x in range(len(ax7) - 1) if (ax7[x:x + len(sub)] == sub).all()]
            [idx8.append(x + 1) for x in range(len(ax8) - 1) if (ax8[x:x + len(sub)] == sub).all()]

            run_con_5 = np.zeros(shape=ax5.shape)
            run_con_6 = np.zeros(shape=ax6.shape)
            run_con_7 = np.zeros(shape=ax7.shape)
            run_con_8 = np.zeros(shape=ax8.shape)

            run_con_5[idx5] = 1
            run_con_6[idx6] = 1
            run_con_7[idx7] = 1
            run_con_8[idx8] = 1

        [idx3.append(x + 1) for x in range(len(ax3) - 1) if (ax3[x:x + len(sub)] == sub).all()]
        [idx4.append(x + 1) for x in range(len(ax4) - 1) if (ax4[x:x + len(sub)] == sub).all()]

        run_con_3 = np.zeros(shape=ax3.shape)
        run_con_4 = np.zeros(shape=ax4.shape)

        run_con_3[idx3] = 1
        run_con_4[idx4] = 1

    [idx1.append(x + 1) for x in range(len(ax1) - 1) if (ax1[x:x + len(sub)] == sub).all()]
    [idx2.append(x + 1) for x in range(len(ax2) - 1) if (ax2[x:x + len(sub)] == sub).all()]

    run_con_1 = np.zeros(shape=ax1.shape)
    run_con_2 = np.zeros(shape=ax2.shape)

    run_con_1[idx1] = 1
    run_con_2[idx2] = 1

    return np.logical_or(np.logical_or(np.logical_or(run_con_1, run_con_2).astype(int),
                                       np.logical_or(run_con_3, run_con_4).astype(int)),
                         np.logical_or(np.logical_or(run_con_5, run_con_6).astype(int),
                                       np.logical_or(run_con_7, run_con_8).astype(int)))


from os import listdir
from os.path import join
import pandas as pd
import numpy as np
import random as rd
import sys
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

modality = ['perception', 'imagery']
trial_type = ['single', 'average']
# category = ['face', 'place']
controls = ['positive', 'con_1', 'con_2', 'con_3', 'con_4', 'con_5', 'con_6', 'null']

tsv_dir = sys.argv[1]
output_dir = sys.argv[2]
# tsv_dir = '/Users/carricarte/PhD/Debugging/sub-01/behdat'
# output_dir = '/Users/carricarte/Desktop'

svc = SVC()
c = [1]
cv = 'run'
param_grid = dict(C=c)

df = pd.DataFrame()
cf = pd.DataFrame()

tsvfiles = []
[tsvfiles.append(t) for t in listdir(tsv_dir) if "run" in t and t.endswith(".tsv") and "._" not in t]
tsvfiles.sort()

for c in controls:

    confounder = np.array([])
    all_events_of_interest = np.array([])

    for i, tsv in enumerate(tsvfiles):
        #  print(tsv)
        datainfo = pd.read_table(join(tsv_dir, tsv))
        events = np.array(datainfo['trial_type'])
        img_paris = np.array(events == 'img_paris').astype(int)
        img_berlin = np.array(events == 'img_berlin').astype(int)
        img_obama = np.array(events == 'img_obama').astype(int)
        img_merkel = np.array(events == 'img_merkel').astype(int)
        seen_paris = np.array(events == 'seen_paris').astype(int)
        seen_berlin = np.array(events == 'seen_berin').astype(int)
        seen_obama = np.array(events == 'seen_obama').astype(int)
        seen_merkel = np.array(events == 'seen_merkel').astype(int)

        img_berlin[img_berlin == 1] = 2
        img_obama[img_obama == 1] = 3
        img_merkel[img_merkel == 1] = 4
        seen_paris[seen_paris == 1] = 5
        seen_berlin[seen_berlin == 1] = 6
        seen_obama[seen_obama == 1] = 7
        seen_merkel[seen_merkel == 1] = 8

        img_paris = np.reshape(img_paris, (1, len(img_paris)))
        img_obama = np.reshape(img_obama, (1, len(img_obama)))
        img_berlin = np.reshape(img_berlin, (1, len(img_berlin)))
        img_merkel = np.reshape(img_merkel, (1, len(img_merkel)))
        seen_paris = np.reshape(seen_paris, (1, len(seen_paris)))
        seen_berlin = np.reshape(seen_berlin, (1, len(seen_berlin)))
        seen_obama = np.reshape(seen_obama, (1, len(seen_obama)))
        seen_merkel = np.reshape(seen_merkel, (1, len(seen_merkel)))

        events_of_interest = np.ndarray.sum(np.concatenate((img_paris, img_obama, img_berlin, img_merkel,
                                                            seen_paris, seen_berlin, seen_obama, seen_merkel), axis=0),
                                            axis=0)
        events_of_interest = events_of_interest[events_of_interest != 0]
        if c == 'positive':
            run_con = events_of_interest
        elif c == 'con_1' or c == 'con_2' or c == 'con_3' or c == 'con_4':
            run_con = get_confounder(c)
        elif c == 'con_5':
            run_con = range(1, len(events_of_interest) + 1)
        elif c == 'con_6':
            run_con = np.zeros(shape=events_of_interest.shape) + i + 1
        elif c == 'null':
            run_con = np.random.rand(len(events_of_interest))

        confounder = np.concatenate((confounder, run_con), axis=0)
        confounder[confounder == 0] = 2
        all_events_of_interest = np.concatenate((all_events_of_interest, events_of_interest), axis=0).astype(int)

    print(confounder)
    print(all_events_of_interest)

    # analysis
    acc = []
    pvals = []
    for m in modality:
        # for t in trial_type:
        # for cat in category:

            if m == 'perception': # and cat == 'face':
                # boolean_ind = all_events_of_interest > 4
                c1, c2 = [5, 6]
                c3, c4 = [7, 8]
            elif m == 'imagery': # and cat == 'face':
                # boolean_ind = all_events_of_interest < 5
                c1, c2 = [1, 2]
                c3, c4 = [3, 4]
            # if m == 'perception' and cat == 'place':
            #     # boolean_ind = all_events_of_interest > 4
            #     c1, c2 = [5, 6]
            # elif m == 'imagery' and cat == 'place':
            #     boolean_ind = all_events_of_interest < 5
            #     c1, c2 = [1, 2]
            # label = all_events_of_interest[boolean_ind]
            y = np.zeros(shape=all_events_of_interest.shape)

            # y[np.logical_or(label == c1, label == c2)] = 2
            y[all_events_of_interest == c1] = 1
            y[all_events_of_interest == c2] = 1
            y[all_events_of_interest == c3] = 2
            y[all_events_of_interest == c4] = 2
            X = confounder[y != 0]
            y = y[y != 0]

            # if t == "average":
            #
            #     _acc = []
            #     for i in range(0, 100):
            #
            #         Xc1 = X[y == 1]
            #         Xc2 = X[y == 2]
            #         Xc1 = Xc1[np.random.permutation(Xc1.shape[0])]
            #         Xc2 = Xc2[np.random.permutation(Xc2.shape[0])]
            #         samples = Xc1.shape[0]
            #         if samples == 18:
            #             parts = 6
            #         elif samples == 20:
            #             parts = 5
            #         elif samples == 24:
            #             parts = 6
            #         elif samples == 72:
            #             parts = 8
            #         elif samples == 80:
            #             parts = 8
            #         elif samples == 96:
            #             parts = 8
            #         mXc1 = np.mean(np.split(Xc1, parts), axis=1)
            #         mXc2 = np.mean(np.split(Xc2, parts), axis=1)
            #         X_train = np.concatenate([mXc1, mXc2], axis=0)
            #         X_train = X_train.reshape(-1, 1)
            #         y_train = np.ones(shape=X_train.shape[0]).astype(int)
            # #
            # #         # scaler = preprocessing.StandardScaler().fit(X_mean)
            # #         # X_mean = scaler.transform(X_mean)
            #         y_train[mXc1.shape[0]:] = 2
            # #
            #         grid_avg = GridSearchCV(svc, param_grid, cv=int(X_train.shape[0] / 2), scoring='accuracy')
            #         grid_avg.fit(X_train, y_train)
            #         score = grid_avg.best_score_
            #         _acc.append(score)
            #         effect = np.mean(_acc)
            # #
            # elif t == "single":
            X_train = X.reshape(-1, 1)
            y_train = y

            acc_ = []
            repetitions = int(len(y) / len(tsvfiles))
            for p in range(len(tsvfiles)):
                mask_train = np.ones(len(y), dtype=bool)
                mask_train[p * repetitions:(p + 1) * repetitions] = False
                # print(mask_train)
                X_train = X[mask_train].reshape(-1, 1)
                y_train = y[mask_train]
                X_test = X[mask_train == False].reshape(-1, 1)
                y_test = y[mask_train == False]
                svc.fit(X_train, y_train)
                acc_.append(svc.score(X_test, y_test))
            #
            effect = np.mean(acc_)

            # scaler = preprocessing.StandardScaler().fit(X)
            # X = scaler.transform(X)
            # grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy')
            # grid.fit(X_train, y_train)
            # effect = grid.best_score_
            #
            acc.append(effect)

            # Compute the null distribution of the classifier accuracy
            if c == 'null':
                acc_ = []
                null_estim = []
                for i in range(0, 1000):
                    rd.shuffle(y_train)
                    # grid.fit(X_train, y_train)
                    # null_estim.append(grid.best_score_)
                    _acc = []
                    for p in range(len(tsvfiles)):
                        mask_train = np.ones(len(y), dtype=bool)
                        mask_train[p * repetitions:(p + 1) * repetitions] = False
                        # print(mask_train)
                        X_train = X[mask_train].reshape(-1, 1)
                        y_train = y[mask_train]
                        X_test = X[mask_train == False].reshape(-1, 1)
                        y_test = y[mask_train == False]
                        svc.fit(X_train, y_train)
                        _acc.append(svc.score(X_test, y_test))
                    acc_.append(np.mean(_acc))
                    null_estim = acc_

                null_distribution = 1.0 * np.array(null_estim)
                pvals.append(np.array(null_distribution >= effect).sum() / len(null_distribution))

            # acc.append(np.mean(acc_))

                # if m == 'perception' and t == 'average':
                #     line_x = [effect, effect]
                #     plt.hist(null_distribution, bins=50)
                #     line_y = np.array(plt.ylim())
                #     plt.plot(line_x, line_y, color='red')
                #     plt.xlabel("Accuracy")
                #     plt.ylabel("Frequency")
                #     plt.savefig(join(output_dir, 'acc_control.png'.format(t)))

    df[c] = acc
    cf[c] = X.T
df['null_p'] = pvals
df['modality'] = ['perception', 'imagery']
df['trial_type'] = ['single', 'single']
df.to_csv(join(output_dir, 'acc_confounder_category_cv-{}.tsv'.format(cv)), sep='\t', index=True)

