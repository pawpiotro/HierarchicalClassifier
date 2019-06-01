import os

from bunch import bunch


# Metody odpowiedzialne za wyliczanie danych statystycznych związanych
# z podanym zbiorem danych lub wynikiem klasyfikacji
# Także metody modyfikujące w zdefiniowany sposób dostarczony zbiór danych
def get_examples_count(result, is_positive):
    count = 0
    for x in result:
        if x == is_positive:
            count += 1

    return count


def get_examples_filenames(dataset, is_positive):
    examples_filenames = []
    for i in range(0, len(dataset.target)):
        if dataset.target[i] == is_positive:
            examples_filenames.append(reduce_filename(dataset.filenames[i]))

    return examples_filenames


def get_categories_examples_count(examples):
    res = {}

    for example_filename in examples:
        category = example_filename[0:example_filename.find(os.path.sep)]
        if category in res:
            res[category] += 1
        else:
            res[category] = 1

    return res


def add_example(src_dataset, dst_dataset, example_idx):
    dst_dataset.data.append(src_dataset.data[example_idx])
    dst_dataset.filenames.append(reduce_filename(
                                        src_dataset.filenames[example_idx]))

    target_name = src_dataset.target_names[src_dataset.target[example_idx]]
    if (target_name in dst_dataset.target_names):
        idx = dst_dataset.target_names.index(target_name)
        dst_dataset.target.append(idx)
    else:
        dst_dataset.target_names.append(target_name)
        idx = len(dst_dataset.target_names) - 1
        dst_dataset.target.append(idx)


def intersect_datasets(datasets, real_res):
    positive_examples = bunch()
    negative_examples = bunch()

    for i in range(0, len(real_res)):
        if real_res[i] == 1:
            add_example(datasets, positive_examples, i)
        else:
            add_example(datasets, negative_examples, i)

    return (negative_examples, positive_examples)


def reduce_filename(filename):
    second_to_last_slash_idx = filename.rfind(os.path.sep, 0,
                                              filename.rfind(os.path.sep))
    next_idx = second_to_last_slash_idx + 1
    reduced_filename = filename[next_idx:]

    return reduced_filename
