# %% [markdown]
# # PROJET : Analyse de données · régression linéaire, ANOVA, khi-2
# Nom, prénom : ......................   Groupe : ............
# Bibliothèques : pandas, numpy, scipy, matplotlib, seaborn (statsmodels autorisé en option).
# Complétez les zones "# TODO". Commentez vos choix et interprétez TOUS vos résultats par écrit.
# %%
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
ALPHA = 0.05

# %% PARTIE 0 : chargement et nettoyage (2 pts)
df = pd.read_csv("donnees_projet.csv")
# TODO 0.1 : dimensions, types, valeurs manquantes par variable
print(df.shape)                 # (lignes, colonnes)
print(df.dtypes)                # type de chaque variable
print(df.isna().sum())          # valeurs manquantes par variable
# TODO 0.2 : repérer une valeur aberrante dans heures_etude (describe, boxplot). Que faites-vous ? Justifiez.
print(df["heures_etude"].describe())
print(df.nlargest(5, "heures_etude")[["id", "heures_etude", "note_stat"]])

plt.figure(figsize=(6, 3))
sns.boxplot(x=df["heures_etude"])
plt.title("Heures d'étude par semaine (avant nettoyage)")
plt.xlabel("heures d'étude / semaine")
plt.show()

# Décision : 95 h/semaine est irréaliste (≈ 13,5 h par jour, 7 jours sur 7).
# Probable erreur de saisie (9,5 ?), mais on ne peut pas le prouver : on met la valeur à NaN
# plutôt que de la corriger au hasard. On garde donc l'étudiant pour les autres variables.
df["heures_etude_brut"] = df["heures_etude"]          # on garde une trace de l'original
df.loc[df["heures_etude"] > 80, "heures_etude"] = np.nan
# TODO 0.3 : construire le tableau d'analyse "d" sans valeurs manquantes sur les variables utilisées
# heures_sommeil n'est utilisée dans aucune analyse demandée : on ne s'en sert pas pour filtrer.
vars_utilisees = ["filiere", "sexe", "logement", "assiduite",
                  "moyenne_bac", "heures_etude", "note_stat"]
d = df.dropna(subset=vars_utilisees).copy()
print(len(df), "->", len(d), "lignes conservées")

# La valeur 95 h/semaine (étudiant n°18) est irréaliste : elle représente plus de 13 h d'étude par jour, sans repos. Elle est très éloignée des autres observations (le maximum suivant est 24,9 h) et fausserait la droite de régression. Comme on ne peut pas prouver qu'il s'agit de 9,5 h, je la remplace par une valeur manquante plutôt que de la corriger. Les 6 valeurs manquantes sont retirées de d, qui contient 293 étudiants.
# Le fichier contient 300 étudiants et 10 variables. Après le nettoyage, 293 étudiants
# restent dans d : 6 heures d'étude déjà manquantes et l'observation aberrante sont exclues.

# %% PARTIE 1 : analyse descriptive (2 pts)
# TODO 1.1 : moyenne, écart-type, min, max de moyenne_bac, heures_etude, note_stat
print("Analyse descriptive des variables quantitatives :")
quanti = ["moyenne_bac", "heures_etude", "note_stat"]
print(d[quanti].agg(["mean", "std", "min", "max"]).round(2))
# En moyenne, les étudiants ont 12,89/20 au bac, étudient 8,95 h par semaine
# et obtiennent 11,08/20 en statistique ; les écarts-types décrivent leur dispersion.

# TODO 1.2 : moyenne de note_stat par filière ; effectifs de chaque modalité qualitative
print("Moyenne de la note en statistique par filière :")
print(d.groupby("filiere")["note_stat"].mean().round(2))
# La moyenne est la plus élevée en Informatique (12,75), puis en Biologie (10,95),
# et la plus faible en Chimie (9,90) ; ces écarts seront testés par l'ANOVA.
for col in ["filiere", "sexe", "logement", "assiduite", "reussite"]:
    print(d[col].value_counts(), "\n")


# TODO 1.3 : 2 graphiques commentés (histogramme et boxplot par exemple)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(d["note_stat"], bins=15, kde=True, ax=ax[0])
ax[0].set(title="Distribution de la note en statistique",
          xlabel="note (/20)", ylabel="effectif")
sns.boxplot(data=d, x="filiere", y="note_stat", ax=ax[1])
ax[1].set(title="Note en statistique selon la filière",
          xlabel="filière", ylabel="note (/20)")
plt.tight_layout()
plt.show()
# L'histogramme est centré autour de 11/20. Le boxplot suggère des notes plus
# élevées en Informatique qu'en Biologie et en Chimie ; le test ANOVA vérifie cela.


# %% PARTIE 2 : régression linéaire simple, note_stat ~ heures_etude (6 pts)
x = d["heures_etude"]
y = d["note_stat"]
n = len(d)

# TODO 2.1 : nuage de points. La relation semble-t-elle linéaire ?
plt.figure(figsize=(6, 4))
sns.scatterplot(x=x, y=y)
plt.title("Note en statistique selon les heures d'étude")
plt.xlabel("heures d'étude / semaine")
plt.ylabel("note (/20)")
plt.show()
# Le nuage montre une tendance croissante assez proche d'une droite, avec une
# dispersion des notes autour de cette tendance.

# TODO 2.2 : coefficient de corrélation r et son test
r, p_r = stats.pearsonr(x, y)
print(f"r = {r:.3f}, p-value = {p_r:.2e}")
# r = 0,677 indique une association linéaire positive assez forte. Comme p < 0,05,
# on rejette l'hypothèse d'une corrélation linéaire nulle.

# TODO 2.3 : estimer la droite (stats.linregress) ; écrire l'équation ; interpréter pente et ordonnée
res = stats.linregress(x, y)
print(f"note = {res.intercept:.2f} + {res.slope:.3f} x heures_etude")
# La pente (0,544) signifie qu'une heure d'étude hebdomadaire supplémentaire est
# associée à environ 0,54 point de plus en moyenne. L'ordonnée 6,20 est la note
# estimée à zéro heure, valeur hors de la plage observée et donc peu interprétable.


# TODO 2.4 : R², p-value de la pente, intervalle de confiance de la pente
r2 = res.rvalue ** 2
t_crit = stats.t.ppf(1 - ALPHA / 2, df=n - 2)
ic = (res.slope - t_crit * res.stderr, res.slope + t_crit * res.stderr)
print(f"R² = {r2:.3f}")
print(f"p-value de la pente = {res.pvalue:.2e}")
print(f"IC95% de la pente = [{ic[0]:.3f} ; {ic[1]:.3f}]")
# Le modèle explique 45,9 % de la variabilité des notes. L'IC95 % de la pente
# [0,476 ; 0,612] n'inclut pas zéro, ce qui confirme une pente positive.


# TODO 2.5 : retrouver pente et ordonnée "à la main" (formules du cours) et comparer
# b1 = Σ(xi - x̄)(yi - ȳ) / Σ(xi - x̄)²      b0 = ȳ - b1 · x̄
x_bar, y_bar = x.mean(), y.mean()
b1 = ((x - x_bar) * (y - y_bar)).sum() / ((x - x_bar) ** 2).sum()
b0 = y_bar - b1 * x_bar
print(f"à la main  : b1 = {b1:.4f}, b0 = {b0:.4f}")
print(f"linregress : b1 = {res.slope:.4f}, b0 = {res.intercept:.4f}")
# Les deux méthodes donnent les mêmes coefficients : le calcul manuel vérifie
# donc le résultat de stats.linregress.


# TODO 2.6 : prédire la note pour 10 h d'étude par semaine ; peut-on prédire pour 40 h ? pourquoi ?
for h in (10, 40):
    print(f"{h} h/semaine -> note prédite = {res.intercept + res.slope * h:.2f}")
print("Plage observée des heures d'étude :", x.min(), "à", x.max())
# La prédiction à 10 h (11,65/20) est dans la plage observée. À 40 h, le modèle
# prédit 27,97/20, résultat impossible : 40 h dépasse le maximum observé (24,9 h),
# et cette extrapolation ne doit pas être interprétée comme une prédiction fiable.


# TODO 2.7 : résidus : graphique résidus vs ajustées, QQ-plot, test de Shapiro ; conditions respectées ?
fitted = res.intercept + res.slope * x
resid = y - fitted

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.scatterplot(x=fitted, y=resid, ax=ax[0])
ax[0].axhline(0, color="red", ls="--")
ax[0].set(title="Résidus vs valeurs ajustées",
          xlabel="note ajustée", ylabel="résidu")
stats.probplot(resid, dist="norm", plot=ax[1])
ax[1].set_title("QQ-plot des résidus")
plt.tight_layout()
plt.show()

W, p_sw = stats.shapiro(resid)
print(f"Shapiro-Wilk : W = {W:.3f}, p-value = {p_sw:.3f}")
# Le QQ-plot suit globalement la droite et Shapiro donne p = 0,490 : on ne
# rejette pas la normalité des résidus. Le nuage résidus-ajustées ne montre pas
# de structure nette ; linéarité et variance constante paraissent plausibles.
# Ces diagnostics ne vérifient toutefois pas l'indépendance des observations.


# BONUS 2.8 : régression multiple (heures_etude + moyenne_bac) avec scikit-learn ou statsmodels ; comparer les R²
# Modèle : note = b0 + b1·heures_etude + b2·moyenne_bac
X = np.column_stack([np.ones(n), d["heures_etude"].values, d["moyenne_bac"].values])
beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)      # moindres carrés
p = X.shape[1]                                           # nombre de paramètres (3)

res_multi = y.values - X @ beta
sce = (res_multi ** 2).sum()                  # somme des carrés des résidus
sct = ((y - y.mean()) ** 2).sum()             # somme des carrés totale
r2_multi = 1 - sce / sct
r2_aj_multi = 1 - (sce / (n - p)) / (sct / (n - 1))
r2_aj_simple = 1 - (1 - r2) * (n - 1) / (n - 2)

# erreurs standard et p-values des coefficients
s2 = sce / (n - p)
se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
p_vals = 2 * stats.t.sf(np.abs(beta / se), df=n - p)

for nom, b, p_v in zip(["constante", "heures_etude", "moyenne_bac"], beta, p_vals):
    print(f"{nom:13s}: coef = {b:.3f}, p-value = {p_v:.2e}")
print(f"R² simple = {r2:.3f} (ajusté {r2_aj_simple:.3f})")
print(f"R² multiple = {r2_multi:.3f} (ajusté {r2_aj_multi:.3f})")
# Ajouter la moyenne au bac fait passer le R² ajusté de 0,457 à 0,507.
# Dans ce modèle, les deux pentes (heures et moyenne_bac) sont significatives
# à 5 % ; leur association avec la note est estimée en contrôlant l'autre variable.


# %% PARTIE 3 : ANOVA à un facteur, note_stat ~ filiere (5 pts)
# TODO 3.1 : H0 et H1 ; boxplot de la note par filière
# H0 : mu_Biologie = mu_Chimie = mu_Informatique
# H1 : au moins une moyenne diffère des autres
plt.figure(figsize=(6, 4))
sns.boxplot(data=d, x="filiere", y="note_stat")
plt.title("Note en statistique selon la filière")
plt.xlabel("filière"); plt.ylabel("note (/20)")
plt.show()


# TODO 3.2 : vérifier les conditions : normalité par groupe (Shapiro), égalité des variances (Levene)
groupes = {f: g["note_stat"].values for f, g in d.groupby("filiere")}
for f, v in groupes.items():                       # normalité par groupe
    W, p = stats.shapiro(v)
    print(f"Shapiro {f:13s} : W = {W:.3f}, p = {p:.3f}")
stat_lev, p_lev = stats.levene(*groupes.values())  # égalité des variances
print(f"Levene : stat = {stat_lev:.3f}, p = {p_lev:.3f}")
# Les trois tests de Shapiro (p = 0,900 ; 0,117 ; 0,467) et celui de Levene
# (p = 0,646) ne signalent pas de violation de la normalité ni de l'égalité
# des variances au seuil de 5 % ; l'indépendance reste une hypothèse de l'étude.


# TODO 3.3 : calculer SC_entre, SC_intra, CM, F, ddl à la main puis comparer avec stats.f_oneway
y = d["note_stat"]
N, k = len(y), len(groupes)
moy_gen = y.mean()
sc_entre = sum(len(v) * (v.mean() - moy_gen) ** 2 for v in groupes.values())
sc_intra = sum(((v - v.mean()) ** 2).sum() for v in groupes.values())
ddl_entre, ddl_intra = k - 1, N - k
cm_entre = sc_entre / ddl_entre
cm_intra = sc_intra / ddl_intra
F = cm_entre / cm_intra
print(f"SC_entre = {sc_entre:.2f}, SC_intra = {sc_intra:.2f}, SC_totale = {sc_entre + sc_intra:.2f}")
print(f"CM_entre = {cm_entre:.2f}, CM_intra = {cm_intra:.2f}, F = {F:.2f}, ddl = ({ddl_entre}, {ddl_intra})")
F_lib, p_lib = stats.f_oneway(*groupes.values())
print(f"f_oneway : F = {F_lib:.2f}, p = {p_lib:.2e}")
# Le calcul manuel retrouve la statistique de scipy : F(2, 290) = 29,52.


# TODO 3.4 : F critique (stats.f.ppf) ; décision par comparaison F / F critique ET par la p-value
F_crit = stats.f.ppf(1 - ALPHA, ddl_entre, ddl_intra)
p_val = stats.f.sf(F, ddl_entre, ddl_intra)
print(f"F critique = {F_crit:.3f} ; F = {F:.2f} -> {'rejet de H0' if F > F_crit else 'H0 non rejetée'}")
print(f"p-value = {p_val:.2e} -> {'rejet de H0' if p_val < ALPHA else 'H0 non rejetée'}")
# F dépasse F critique (29,52 > 3,027) et p < 0,05 : on rejette H0.
# Au moins une moyenne de filière diffère ; l'ANOVA seule ne dit pas lesquelles.


# TODO 3.5 : taille d'effet η² ; interprétation
eta2 = sc_entre / (sc_entre + sc_intra)
print(f"eta² = {eta2:.3f}")
# eta² = 0,169 : environ 16,9 % de la variabilité des notes observées est
# associée aux différences entre filières ; c'est une taille d'effet notable.


# TODO 3.6 : comparaisons par paires (stats.tukey_hsd) : quelles filières diffèrent ?
tuk = stats.tukey_hsd(*groupes.values())
print(list(groupes))       # ordre : 0 = Biologie, 1 = Chimie, 2 = Informatique
print(tuk)
# Tukey conclut que chaque paire diffère (Biologie-Chimie p = 0,012 ;
# les deux comparaisons avec Informatique p < 0,001). Le classement des
# moyennes est donc Informatique > Biologie > Chimie.


# BONUS 3.7 : test non paramétrique (stats.kruskal) ; conclusion identique ?
H, p_kw = stats.kruskal(*groupes.values())
print(f"Kruskal-Wallis : H = {H:.2f}, p = {p_kw:.2e}")
# Kruskal-Wallis est aussi significatif (p < 0,05) : la conclusion générale
# d'une différence entre filières est cohérente avec l'ANOVA.

# %% PARTIE 4 : khi-2 d'indépendance (5 pts)
# Deux couples à tester : (logement, assiduite) et (sexe, filiere)
def test_khi2(var_ligne, var_col):
    print("=" * 60)
    print(f"Test d'indépendance : {var_ligne} x {var_col}")
    print(f"H0 : {var_ligne} et {var_col} sont indépendantes")
    print(f"H1 : {var_ligne} et {var_col} sont liées")


# Pour CHAQUE couple :
    # TODO 4.1 : H0, H1 ; tableau croisé observé (pd.crosstab)
    obs = pd.crosstab(d[var_ligne], d[var_col])
    print("\nEffectifs observés :\n", obs)
    N = obs.values.sum()


    # TODO 4.2 : tableau des effectifs théoriques calculés avec la formule du cours (ligne x colonne / total)
    theo = pd.DataFrame(np.outer(obs.sum(axis=1), obs.sum(axis=0)) / N,
                        index=obs.index, columns=obs.columns)
    print("\nEffectifs théoriques :\n", theo.round(2))
    # Sous H0, chaque effectif attendu est le produit du total de sa ligne et
    # du total de sa colonne, divisé par N.


    # TODO 4.3 : contribution de chaque cellule (O-E)²/E, puis le χ² total
    contrib = (obs - theo) ** 2 / theo
    chi2 = contrib.values.sum()
    print("\nContributions (O-E)²/E :\n", contrib.round(2))
    print(f"\nχ² = {chi2:.3f}")
    # Le χ² additionne les contributions de toutes les cellules ; une grande
    # contribution signale une différence importante entre observé et attendu.


    # TODO 4.4 : ddl, χ² critique, p-value ; comparer avec stats.chi2_contingency
    ddl = (obs.shape[0] - 1) * (obs.shape[1] - 1)
    chi2_crit = stats.chi2.ppf(1 - ALPHA, ddl)
    p_val = stats.chi2.sf(chi2, ddl)
    chi2_lib, p_lib, ddl_lib, _ = stats.chi2_contingency(obs)
    print(f"ddl = {ddl}, χ² critique = {chi2_crit:.3f}, p-value = {p_val:.2e}")
    print(f"chi2_contingency : χ² = {chi2_lib:.3f}, p = {p_lib:.2e}, ddl = {ddl_lib}")
    # Les résultats manuels et ceux de scipy coïncident pour ces tableaux.


    # TODO 4.5 : vérifier la condition E >= 5 ; décision ; conclusion en une phrase
    print(f"E minimum = {theo.values.min():.2f} -> condition E >= 5 "
          f"{'respectée' if theo.values.min() >= 5 else 'NON respectée'}")
    print("Décision :", "on rejette H0" if chi2 > chi2_crit else "on ne rejette pas H0")
    # Pour les deux tableaux, tous les effectifs attendus dépassent 5 ; le
    # résultat du test est donc interprétable avec cette approximation.


    # TODO 4.6 : V de Cramér ; graphique en barres empilées en pourcentages
    v = np.sqrt(chi2 / (N * (min(obs.shape) - 1)))
    print(f"V de Cramér = {v:.3f}")
    # V décrit l'intensité de l'association (sans sens causal) : il vaut 0,262
    # pour logement-assiduité et 0,093 pour sexe-filière dans ces données.
    pct = pd.crosstab(d[var_ligne], d[var_col], normalize="index") * 100
    pct.plot(kind="bar", stacked=True, figsize=(6, 4))
    plt.title(f"{var_col} selon {var_ligne} (en %)")
    plt.xlabel(var_ligne); plt.ylabel("pourcentage")
    plt.xticks(rotation=0)
    plt.legend(title=var_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

test_khi2("logement", "assiduite")
test_khi2("sexe", "filiere")
# Logement-assiduité : χ²(4) = 40,276, p < 0,001 ; on rejette H0 et observe
# une association (V = 0,262). Sexe-filière : χ²(2) = 2,549, p = 0,280 ;
# on ne rejette pas H0 et les données ne montrent pas d'association nette.
# Question de réflexion : pourquoi un khi-2 significatif ne prouve-t-il pas une relation de cause à effet ?
# Un test d'indépendance mesure une association, pas l'effet d'une intervention.
# Des variables confondantes, la sélection des participants ou le hasard peuvent
# expliquer cette association ; il ne donne ni ordre temporel ni preuve causale.

# %% PARTIE 5 : synthèse (clarté, 0 pt ici, évalué dans le rapport)
# Rédigez dans le rapport une conclusion de 10 lignes reliant les trois méthodes.
# Le nettoyage conserve 293 étudiants après exclusion des valeurs manquantes utiles à l'analyse.
# La régression met en évidence une association positive entre heures d'étude et note.
# Une heure d'étude hebdomadaire supplémentaire correspond à environ 0,54 point estimé.
# Le modèle simple explique 45,9 % de la variabilité des notes.
# Les diagnostics ne signalent pas de violation majeure des conditions de la régression.
# L'ANOVA montre une différence statistiquement significative entre les filières.
# Tukey situe Informatique au-dessus de Biologie, puis de Chimie.
# Le khi-deux détecte une association entre le logement et l'assiduité.
# En revanche, il ne détecte pas d'association significative entre sexe et filière.
# Ces résultats décrivent des associations dans l'échantillon et ne prouvent pas une causalité.
