import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv


class Node:
    def __init__(self, index=None, value=None, gauche=None, droite=None, is_terminal=False, prediction=None):
        self.index = index
        self.value = value
        self.gauche = gauche
        self.droite = droite
        self.is_terminal = is_terminal
        self.prediction = prediction


class CART:
    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None
        self.label_encoder = LabelEncoder()

    def gini_index(self, groups, classes):
        """Calcul du Gini Index de manière plus efficace."""
        n_instances = float(sum([len(group) for group in groups]))  # Total des instances
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue  # Si le groupe est vide, on l'ignore
            score = 0.0
            for class_value in classes:
                # Calculer la probabilité de chaque classe dans ce groupe
                p = [row[-1] for row in group].count(class_value) / size  # Fréquence de chaque classe
                score += p * p
            gini += (1.0 - score) * (size / n_instances)  # Ponderer selon la taille du groupe
        return gini

    def split_data(self, index, value, dataset):
        """Divise les données en deux groupes en fonction de l'index et de la valeur."""
        gauche, droite = [], []
        for row in dataset:
            if row[index] < value:
                gauche.append(row)
            else:
                droite.append(row)
        return gauche, droite

    def best_split(self, dataset):
        """Trouve la meilleure division des données selon le Gini."""
        class_values = list(set(row[-1] for row in dataset))  # Les classes possibles
        best_index, best_value, best_score, best_groups = 999, 999, 999, None

        for index in range(len(dataset[0]) - 1):  # Ignore la classe (dernier élément)
            for row in dataset:
                groups = self.split_data(index, row[index], dataset)

                # Débogage pour vérifier les groupes et leur contenu
                print(f"Index: {index}, Value: {row[index]}, Groups: {groups}")

                gini = self.gini_index(groups, class_values)

                # Affichage du gini pour chaque division
                print(f"Gini: {gini}")

                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups

        return {'index': best_index, 'value': best_value, 'groups': best_groups}

    def to_terminal(self, group):
        """Retourne la classe prédominante dans le groupe."""
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def split(self, node, depth):
        """Divise un nœud en deux enfants."""
        gauche, droite = node['groups']
        del (node['groups'])
        if not gauche or not droite:
            prediction = self.to_terminal(gauche + droite)
            return Node(is_terminal=True, prediction=prediction)

        if depth >= self.max_depth:
            return Node(is_terminal=True, prediction=self.to_terminal(gauche)), Node(is_terminal=True,
                                                                                     prediction=self.to_terminal(
                                                                                         droite))

        left_child = None
        if len(gauche) > self.min_size:
            left_child = self.best_split(gauche)
            left_child = self.split(left_child, depth + 1)
        else:
            left_child = Node(is_terminal=True, prediction=self.to_terminal(gauche))

        right_child = None
        if len(droite) > self.min_size:
            right_child = self.best_split(droite)
            right_child = self.split(right_child, depth + 1)
        else:
            right_child = Node(is_terminal=True, prediction=self.to_terminal(droite))

        return Node(node['index'], node['value'], left_child, right_child)

    def build_tree(self, dataset):
        """Construit l'arbre de décision à partir des données."""
        root = self.best_split(dataset)
        self.root = self.split(root, 1)

    def predict(self, node, row):
        """Prédit la classe pour une ligne donnée en suivant l'arbre."""
        if node.is_terminal:
            return node.prediction
        elif row[node.index] < node.value:
            return self.predict(node.gauche, row)
        else:
            return self.predict(node.droite, row)

    def fit(self, dataset):
        """Entraîne l'arbre de décision avec les données fournies."""
        # Encoder les labels dans les données
        for row in dataset:
            row[-1] = self.label_encoder.fit_transform([row[-1]])[0]  # Encoder la classe
        self.build_tree(dataset)

    def predict_row(self, row):
        """Prédit la classe d'une seule ligne."""
        prediction = self.predict(self.root, row)
        return self.label_encoder.inverse_transform([prediction])[0]  # Décoder la prédiction

    def predict_dataset(self, dataset):
        """Prédit les classes pour l'ensemble des données."""
        return [self.predict_row(row) for row in dataset]


def load_and_prepare_data(file_path):
    """Charge et prépare les données pour l'algorithme CART."""
    data = pd.read_csv(file_path)

    # Supprimer la colonne ID (inutile pour l'analyse)
    if 'ID' in data.columns:
        data = data.drop(columns=['ID'])

    # Identifier les colonnes catégoriques
    categorical_columns = ['Historique de Crédit', 'Prêt Approuvé']
    label_encoders = {}

    # Encoder les colonnes catégoriques
    for col in categorical_columns:
        if col in data.columns:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])
            label_encoders[col] = encoder

    # Convertir en tableau numpy pour l'entraînement
    dataset = data.values
    return dataset, label_encoders


# Charger et préparer les données
file_path = "donnees.csv"  # Remplacez par le chemin correct
dataset, encoders = load_and_prepare_data(file_path)

# Initialiser et entraîner l'arbre de décision CART
tree = CART(max_depth=5, min_size=10)
tree.fit(dataset)

# Prédictions
predictions = tree.predict_dataset(dataset)

# Afficher les prédictions
for row, prediction in zip(dataset, predictions):
    print(f"Prédiction: {prediction}, Réel: {row[-1]}")
