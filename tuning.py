# tuning.py


import numpy as np

from lda_analyzer import LdaAnalyzer


def hyperparameter_tuning():
    """
    Hyperparamter tuning for author-topic model.
    """
    lda = LdaAnalyzer()
    topics = (25, 50, 75, 100)
    belows = (1, 2, 3, 4, 5)
    aboves = (0.3, 0.4, 0.5, 0.6, 0.7)
    for t in topics:
        for b in belows:
            for a in aboves:
                authors, coherence = lda.get_topics_atm(num_topics=t, no_below=b, no_above=a, verbose=False)
                print('Coherence for t={}, b={}, a={}:\t\t{}'.format(t, b, a, coherence))
                print(authors)
                print()


def hyperparameter_tuning_2():
    # Coherence for t=100, b=3, a=0.3:		0.4698965899917743
    # Coherence for t=100, b=2, a=0.3:		0.49201773809635685
    # Coherence for t=100, b=2, a=0.4:		0.46331219333225176
    # Coherence for t=100, b=2, a=0.5:		0.46342722178815066
    # Coherence for t=75, b=2, a=0.4:		0.45514293172373366
    # Coherence for t=75, b=2, a=0.5:		0.44551576278911453
    # Coherence for t=25, b=1, a=0.6:		0.44955693297379645

    # Coherence for t=200, b=3, a=0.3:		0.5262358075990899
    # Coherence for t=200, b=2, a=0.4:		0.5180206524696847
    # Coherence for t=200, b=2, a=0.3:		0.5273626954449002
    # Coherence for t=150, b=2, a=0.4:		0.5110369610604198
    # Coherence for t=150, b=2, a=0.3:		0.5215788006405994
    lda = LdaAnalyzer()
    params = [
        (200, 3, 0.3),
        (200, 2, 0.4),
        (200, 2, 0.3),
        (150, 2, 0.4),
        (150, 2, 0.3),
    ]
    for t, b, a in params:
        for ti in range(t - 5, t + 5, 3):
            for ai in [a - 0.05, a, a + 0.05]:
                authors, coherence = lda.get_topics_atm(num_topics=ti, no_below=b, no_above=ai, verbose=False)
                print('Coherence for t={}, b={}, a={}:\t\t{}'.format(ti, b, ai, coherence))
                print(authors)
                print()
    topics = (180, 220, 240)
    belows = (2, 3)
    aboves = (0.3, 0.4, 0.5)
    for t in topics:
        for b in belows:
            for a in aboves:
                authors, coherence = lda.get_topics_atm(num_topics=t, no_below=b, no_above=a, verbose=False)
                print('Coherence for t={}, b={}, a={}:\t\t{}'.format(t, b, a, coherence))
                print(authors)
                print()


def hyperparameter_tuning_3():
    # Coherence for t=50, b=2, a=0.4:		0.4354097953145938
    # Coherence for t=220, b=2, a=0.3, alpha=0.31, beta=0.31:		0.6082240558822126

    # Coherence for t=220, b=2, a=0.3, alpha=0.01, eta=0.01:		0.5531857919261015

    # Coherence for t=220, b=2, a=0.3:		0.5455499374264025
    # Coherence for t=145, b=2, a=0.25:		0.5407903732108236
    # Coherence for t=204, b=2, a=0.3:		0.5393914879467542
    # Coherence for t=201, b=2, a=0.25:		0.5421172975905091
    lda = LdaAnalyzer()
    alphas = np.arange(0.01, 1, 0.3).tolist()
    betas = np.arange(0.01, 1, 0.3).tolist()
    params = [
        (220, 2, 0.3),
        (145, 2, 0.25),
        (204, 2, 0.3),
        (201, 2, 0.25),
    ]
    params = [(50, 2, 0.4)]
    for t, b, a in params:
        for alpha in alphas:
            for beta in betas:
                try:
                    authors, coherence = lda.get_topics_atm(num_topics=t, no_below=b, no_above=a, alpha=alpha,
                                                            beta=beta, verbose=False)
                    print('Coherence for t={}, b={}, a={}, alpha={}, beta={}:\t\t{}'
                          .format(t, b, a, alpha, beta, coherence))
                    print(authors)
                    print()
                except:
                    print('Error for t={}, b={}, a={}, alpha={}, beta={}'.format(t, b, a, alpha, beta))
                    print()


if __name__ == '__main__':
    hyperparameter_tuning_3()
