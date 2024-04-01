import numpy as np


def dominates(row, candidate_row):
    return all(r <= c for r, c in zip(row, candidate_row)) and any(r < c for r, c in zip(row, candidate_row))

def update_elitist_archive(variable_archive, objective_archive, new_variable, new_objective):
    dominated_architectures_indices = []
    is_new_architecture_dominated = False
    is_new_architecture_duplicate = False

    for i, (objective, variable) in enumerate(zip(objective_archive, variable_archive)):
        if np.array_equal(objective, new_objective) and np.array_equal(variable, new_variable):
        # if np.array_equal(objective, new_objective):
            is_new_architecture_duplicate = True
            break
        if dominates(objective, new_objective):  # if the new architecture is dominated by any architecture in the archive
            is_new_architecture_dominated = True
            break
        elif dominates(new_objective, objective):  # if the new architecture dominates any architecture in the archive
            dominated_architectures_indices.append(i)

    # If new architecture is not dominated and not a duplicate, remove any dominated architectures from the archive and add the new architecture
    if not is_new_architecture_dominated and not is_new_architecture_duplicate:
        # Remove dominated architectures in reverse order to maintain indices
        for i in sorted(dominated_architectures_indices, reverse=True):
            del objective_archive[i]
            del variable_archive[i]
        # Add new architecture to archive
        objective_archive.append(new_objective)
        variable_archive.append(new_variable)

    return variable_archive, objective_archive

def identify_pareto(scores):
    population_size = scores.shape[0]
    pareto_front = np.ones(population_size, dtype=bool)
    for i in range(population_size):
        for j in range(population_size):
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                pareto_front[i] = 0
                break
    return pareto_front

def remove_dominated_from_archive(variable_archive, objective_archive):
    # Convert the archives to numpy arrays
    objective_archive_np = np.array(objective_archive)
    variable_archive_np = np.array(variable_archive)

    # Use the identify_pareto function to find the indices of the non-dominated individuals
    pareto_front = identify_pareto(objective_archive_np)

    # Use the boolean array to filter the non-dominated individuals
    non_dominated_objective_archive = objective_archive_np[pareto_front].tolist()
    non_dominated_variable_archive = variable_archive_np[pareto_front].tolist()

    return non_dominated_variable_archive, non_dominated_objective_archive

def array_in_arrays(array, array_list):
    return any(np.array_equal(array, candidate) for candidate in array_list)

def mahalanobis(x, data):
    """Compute the Mahalanobis Distance between each row of x and the data"""
    x_minus_mu = x - np.mean(data)
    cov = np.cov(data.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal
