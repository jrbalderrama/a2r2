from numpy import ndarray, random


def lpa(Q: ndarray, δ: float, ε: float, random_state=None) -> ndarray:

    # random generator with seed
    generator = random.default_rng(seed=random_state)

    # differential privacy scale based on the budget
    λ = δ / ε

    # Laplace mechanism applied to whole serie
    Z = generator.laplace(scale=λ, size=Q.size)

    return Q + Z
