import re


def clean_class_operation(cluster):
    cluster_set = set()
    for i in range(0, len(cluster)):
        elem = cluster[i]
        if '.' in elem:
            elem = elem.split(".")[0]
        cluster_set.add(elem)
    return cluster_set


def extract_keywords(cluster):
    keywords_set = set()
    for elem in cluster:
        matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', elem)
        for m in matches:
            keyword = m.group(0)
            if len(keyword) > 1:
                keywords_set.add(keyword)
    return keywords_set


def get_concepts(cluster, keywords):
    concepts = []
    for keyword in keywords:
        matching = [ele for ele in cluster if keyword in ele]
        if len(matching) == len(cluster):
            concepts.append(keyword)
    return concepts


def extract_concepts(source_cl):
    clusters = source_cl.groupby('cluster')['file name'].apply(list)
    concepts = set()
    for cluster in clusters:
        cluster = clean_class_operation(cluster)
        if len(cluster) > 1:
            keywords = extract_keywords(cluster)
            for concept in get_concepts(cluster, keywords):
                concepts.add(concept)

    print(set(concepts))
    print(len(set(concepts)))
    return set(concepts)


def check_duplicates(concepts_list):
    # Check for duplicates
    duplicates = [word for word in set(concepts_list) if concepts_list.count(word) > 1]
    return duplicates
