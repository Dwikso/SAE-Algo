import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import random


class ArbreDecisionCART:
    def __init__(self, profondeur_max, taille_min):
        self.profondeur_max = profondeur_max
        self.taille_min = taille_min
        self.arbre = None

    def calculer_indice_gini(self, groupes, classes):
        """
        Calcule l'indice de Gini pour une division donnée des données.
        """
        total_instances = 0
        for groupe in groupes:
            total_instances += len(groupe)

        gini = 0.0

        for groupe in groupes:
            taille = len(groupe)
            if taille == 0:
                continue
            score = 0.0
            for val_class in classes:
                proportion = 0
                for ligne in groupe:
                    if ligne[-1] == val_class:
                        proportion += 1
                proportion /= taille
                score += proportion ** 2
            gini_groupe = 1.0 - score
            gini += gini_groupe * (taille / total_instances)

        return gini

    def diviser_ensemble(self, index, valeur, ensemble):
        """
        Divise un ensemble de données en deux groupes selon une valeur donnée.
        """
        gauche, droite = [], []
        for ligne in ensemble:
            if ligne[index] < valeur:
                gauche.append(ligne)
            else:
                droite.append(ligne)
        return gauche, droite

    def trouver_meilleure_division(self, ensemble):
        """
        Identifie la meilleure division possible pour un ensemble de données.
        """
        valeurs_classes = []
        for ligne in ensemble:
            if ligne[-1] not in valeurs_classes:
                valeurs_classes.append(ligne[-1])

        meilleur_score = float('inf')
        meilleur_noeud = {'index': None, 'valeur': None, 'groupes': None}

        for index in range(len(ensemble[0]) - 1):
            for ligne in ensemble:
                valeur_courante = ligne[index]
                groupes = self.diviser_ensemble(index, valeur_courante, ensemble)
                score_gini = self.calculer_indice_gini(groupes, valeurs_classes)

                if score_gini < meilleur_score:
                    meilleur_score = score_gini
                    meilleur_noeud['index'] = index
                    meilleur_noeud['valeur'] = valeur_courante
                    meilleur_noeud['groupes'] = groupes

        return meilleur_noeud

    def creer_feuille(self, groupe):
        """
        Crée une feuille terminale contenant la classe la plus fréquente dans un groupe.
        """
        classes = [ligne[-1] for ligne in groupe]

        #Permet de compter la frequences de chaque classe dans la classes
        frequences = {}
        for classe in classes:
            if classe not in frequences:
                frequences[classe] = 0
            frequences[classe] += 1

        #Permet de trouver la classe avec la plus grande fréquences
        classe_frequente = None
        max_frequence = 0
        for classe, frequence in frequences.items():
            if frequence > max_frequence:
                max_frequence = frequence
                classe_frequente = classe

        return classe_frequente

    def diviser_noeud(self, noeud, profondeur):
        """
        Divise un noeud en deux sous-noeuds ou crée des feuilles terminales.
        """
        gauche, droite = noeud['groupes']

        noeud['groupes'] = None

        if not gauche or not droite:
            noeud['gauche'] = noeud['droite'] = self.creer_feuille(gauche + droite)
            return

        if profondeur >= self.profondeur_max:
            noeud['gauche'], noeud['droite'] = self.creer_feuille(gauche), self.creer_feuille(droite)
            return

        if len(gauche) <= self.taille_min:
            noeud['gauche'] = self.creer_feuille(gauche)
        else:
            noeud['gauche'] = self.trouver_meilleure_division(gauche)
            self.diviser_noeud(noeud['gauche'], profondeur + 1)

        if len(droite) <= self.taille_min:
            noeud['droite'] = self.creer_feuille(droite)
        else:
            noeud['droite'] = self.trouver_meilleure_division(droite)
            self.diviser_noeud(noeud['droite'], profondeur + 1)

    def construire_arbre(self, entrainement):
        """
        Construit l'arbre de décision à partir des données d'entraînement.
        """
        self.arbre = self.trouver_meilleure_division(entrainement)
        self.diviser_noeud(self.arbre, 1)

    def predire_classe(self, noeud, ligne):
        """
        Prédit la classe pour une ligne donnée en parcourant l'arbre.
        """
        if ligne[noeud['index']] < noeud['valeur']:
            if isinstance(noeud['gauche'], dict): #Permet de verifier si le noeud gauche est un sous-noeud (dictionnaire) ou alors une feuille
                return self.predire_classe(noeud['gauche'], ligne)
            else:
                return noeud['gauche']
        else:
            if isinstance(noeud['droite'], dict): #Permet de verifier si le noeud droit est un sous-noeud (dictionnaire) ou alors une feuille
                return self.predire_classe(noeud['droite'], ligne)
            else:
                return noeud['droite']

    def ajuster(self, ensemble):
        """
        Ajuste l'arbre de décision aux données d'entraînement.
        """
        self.construire_arbre(ensemble)

    def predire_ligne(self, ligne):
        """
        Prédit la classe pour une seule ligne de données.
        """
        return self.predire_classe(self.arbre, ligne)


class RandomForest:
    def __init__(self, n_arbres=10, profondeur_max=5, taille_min=5):
        self.n_arbres = n_arbres
        self.profondeur_max = profondeur_max
        self.taille_min = taille_min

    def generer_echantillon(self, ensemble):
        """
        Génère un échantillon à partir de l'ensemble d'entraînement.
        """
        return [random.choice(ensemble) for i in range(len(ensemble))]

    def ajuster(self, ensemble):
        """
        Entraîne la forêt en ajustant chaque arbre sur un sous-échantillon.
        """
        self.arbres = []
        for i in range(self.n_arbres):
            sous_ensemble = self.generer_echantillon(ensemble)
            arbre = ArbreDecisionCART(self.profondeur_max, self.taille_min)
            arbre.ajuster(sous_ensemble)
            self.arbres.append(arbre)

    def predire_majoritaire(self, ligne):
        """
        Prédit la classe pour une ligne en combinant les prédictions des arbres.
        """
        predictions = []
        for arbre in self.arbres:
            predictions.append(arbre.predire_ligne(ligne))
        classes_majoritaire = max(set(predictions), key=predictions.count)
        return classes_majoritaire


class Application:
    def __init__(self):
        """Initialise l'interface de l'application et les variables nécessaires."""
        self.root = tk.Tk()
        self.root.title("Prédiction avec Arbre de Décision ou RandomForest")
        self.root.geometry("500x600")
        self.dataset = None
        self.columns = None
        self.model = None
        self.caracteristiques_test = None
        self.classes_reelles = None
        self.algo_choice = tk.StringVar(value="CART")  # Choix par défaut : CART
        self.input_entries = {}

        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        """
        Crée les éléments de l'interface utilisateur.
        """
        label_algo = tk.Label(self.root, text="Sélectionnez l'algorithme :")
        label_algo.pack(pady=5)

        # Menu déroulant pour choisir l'algorithme
        combo_algo = tk.OptionMenu(self.root, self.algo_choice, "CART", "RandomForest")
        combo_algo.pack(pady=5)

        btn_load = tk.Button(self.root, text="Charger un fichier CSV", command=self.charger_donnees)
        btn_load.pack(pady=10)

        # Section pour les entrées utilisateur
        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=10)

        btn_predict = tk.Button(self.root, text="Faire une Prédiction", command=self.effectuer_prediction)
        btn_predict.pack(pady=10)

        btn_quit = tk.Button(self.root, text="Quitter", command=self.root.quit)
        btn_quit.pack(pady=10)

        btn_generate = tk.Button(self.root, text="Générer des Exemples", command=self.afficher_exemples)
        btn_generate.pack(pady=10)

    def pretraiter_donnees(self, filename):
        """
        Prépare les données en les chargeant depuis un fichier CSV.
        Convertit les variables catégoriques en numériques.
        """
        df = pd.read_csv(filename)
        required_columns = ['Historique de Credit', 'Pret Approuve']
        for col in required_columns:
            if col not in df.columns:
                messagebox.showerror("Erreur", f"Le fichier CSV doit contenir la colonne '{col}'")
                return None

        # Convertir les colonnes catégoriques
        df['Historique de Credit'] = df['Historique de Credit'].map({'Bon': 1, 'Mauvais': 0})
        df['Pret Approuve'] = df['Pret Approuve'].map({'Oui': 1, 'Non': 0})
        if 'ID' in df.columns:
            df = df.drop(columns=['ID'])

        return df.values.tolist(), df.columns.tolist()

    def charger_donnees(self):
        """
        Charge un fichier CSV et entraîne le modèle choisi (CART ou RandomForest).
        """
        csv_path = filedialog.askopenfilename(
            title="Sélectionner un fichier CSV",
            filetypes=[("Fichiers CSV", "*.csv")])
        if not csv_path:
            return
        try:
            dataset, self.columns = self.pretraiter_donnees(csv_path)
            ensemble_entrainement, ensemble_test = self.separer_donnees(dataset)

            self.caracteristiques_test = [ligne[:-1] for ligne in ensemble_test]
            self.classes_reelles = [ligne[-1] for ligne in ensemble_test]

            if self.algo_choice.get() == "CART":
                self.model = ArbreDecisionCART(profondeur_max=20, taille_min=1)
            elif self.algo_choice.get() == "RandomForest":
                self.model = RandomForest(n_arbres=100, profondeur_max=10, taille_min=1)

            self.model.ajuster(ensemble_entrainement)

            classes_predites = [self.model.predire_majoritaire(x) if self.algo_choice.get() == "RandomForest"
                                else self.model.predire_ligne(x) for x in
                                self.caracteristiques_test]
            precision = self.calculer_precision(self.classes_reelles, classes_predites) * 100
            messagebox.showinfo("Précision", f"Précision sur l'ensemble de test : {precision:.2f}%")
            messagebox.showinfo("Succès", "Données chargées et modèle entraîné avec succès!")

            self.creer_champs_saisie()
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement des données : {e}")

    def creer_champs_saisie(self):
        """
        Crée des champs de saisie pour les prédictions basés sur les colonnes du jeu de données.
        """
        for widget in self.input_frame.winfo_children():
            widget.destroy()

        self.input_entries.clear()

        for column in self.columns[:-1]:
            label = tk.Label(self.input_frame, text=f"{column}:")
            label.pack(anchor="w", pady=2)
            entry = tk.Entry(self.input_frame)
            entry.pack(fill="x", pady=2)
            self.input_entries[column] = entry

    def separer_donnees(self, dataset, taille_test=100):
        """
        Divise les données en ensemble d'entraînement et de test avec une taille fixe pour le test.
        """
        random.shuffle(dataset)
        ensemble_test = dataset[:taille_test]
        ensemble_entrainement = dataset[taille_test:]
        return ensemble_entrainement, ensemble_test

    def calculer_precision(self, valeurs_reelles, valeurs_predites):
        """
        Calcule la précision du modèle.
        """
        compteur_correct = 0
        for i in range(len(valeurs_reelles)):
            if valeurs_reelles[i] == valeurs_predites[i]:
                compteur_correct += 1
        return compteur_correct / len(valeurs_reelles)

    def get_user_inputs(self):
        """
        Récupère les entrées utilisateur pour effectuer une prédiction.
        """
        user_values = []
        for column, entry in self.input_entries.items():
            try:
                value = float(entry.get())
                user_values.append(value)
            except ValueError:
                messagebox.showerror("Erreur", f"Valeur invalide pour {column}. Veuillez entrer un nombre.")
                return None
        return user_values

    def generer_exemples(self, nb_exemples=5):
        """
        Génère des exemples aléatoires pour tester le modèle.
        """
        exemples = []
        for i in range(nb_exemples):
            revenu = random.randint(1000, 10000)
            montant_pret = random.randint(5000, 50000)
            duree_emploi = random.randint(1, 10)
            historique_credit = random.choice([0, 1])
            exemple = [revenu, montant_pret, duree_emploi, historique_credit]
            prediction = self.model.predire_majoritaire(exemple) if self.algo_choice.get() == "RandomForest" \
                else self.model.predire_ligne(exemple)
            exemples.append((exemple, "Accepté" if prediction == 1 else "Refusé"))
        return exemples

    def afficher_exemples(self):
        """
        Affiche les exemples générés et leurs prédictions.
        """
        if self.model is None:
            messagebox.showerror("Erreur", "Le modèle n'est pas entraîné. Veuillez charger un fichier CSV.")
            return

        exemples = self.generer_exemples()
        for exemple, decision in exemples:
            print(f"Exemple : {exemple} -> Décision : {decision}")

    def effectuer_prediction(self):
        """
        Effectue une prédiction à partir des valeurs saisies par l'utilisateur.
        """
        if self.model is None:
            messagebox.showerror("Erreur", "Le modèle n'est pas entraîné. Veuillez charger un fichier CSV.")
            return

        try:
            user_values = self.get_user_inputs()
            if user_values is None:
                return

            prediction = self.model.predire_majoritaire(user_values) if self.algo_choice.get() == "RandomForest" \
                else self.model.predire_ligne(user_values)
            messagebox.showinfo("Prédiction", f"Prêt Approuvé ? : {'Oui' if prediction == 1 else 'Non'}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur : {e}")

Application()