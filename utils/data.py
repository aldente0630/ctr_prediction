import hashlib


def hash_str(string, n_bins):
    return int(hashlib.md5(string.encode('utf8')).hexdigest(), 16) % (n_bins - 1) + 1


def get_feature_to_index(num_feature_names, cat_feature_names, n_categories, use_field):
    if use_field:
        feature_to_index = {feature: i for i, feature in enumerate(num_feature_names | cat_feature_names)}
    else:
        feature_to_index = {feature: i for i, feature in enumerate(num_feature_names)}
        j = 0
        for feature, n in zip(cat_feature_names, n_categories):
            for label in range(n):
                feature_to_index['_'.join([feature, str(label)])] = len(num_feature_names) + j
                j += 1
    return feature_to_index


def dump_libsvm_file(X, y, file, num_feature_names, cat_feature_names, n_categories, use_field=False, decimals=6,
                     use_hash=False, n_bins=1000):
    feature_to_index = get_feature_to_index(num_feature_names, cat_feature_names, n_categories, use_field)
    with open(file, 'w') as f:
        for i, row in X.iterrows():
            parsed_row = str(y.loc[i]) if y is not None else ''
            for feature in num_feature_names:
                index = str(feature_to_index[feature])
                field = ''.join([index, ':']) if use_field else ''
                parsed_row = ''.join(
                    [parsed_row, ' ', index, ':', field, str(round(row[feature], decimals))])
            for feature, n in zip(cat_feature_names, n_categories):
                if use_field:
                    field = feature_to_index[feature]
                    index = int(row[feature])
                    index = hash_str(str(index), n_bins) if use_hash and n > n_bins else index
                    parsed_row = ''.join([parsed_row, ' ', str(field), ':', str(index), ':1'])
                else:
                    index = int(row[feature])
                    index = hash_str(str(index), n_bins) if use_hash and n > n_bins else index
                    index = feature_to_index['_'.join([feature, str(index)])]
                    parsed_row = ''.join([parsed_row, ' ', str(index), ':1'])
            f.write(parsed_row.lstrip() + '\n')
