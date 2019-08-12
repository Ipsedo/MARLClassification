# MARLClassification

from this [article](https://arxiv.org/abs/1905.04835)

## Description
Essai de reproduction des résultats de l'article cité ci-dessus. Cet article porte sur de la classification d'image utilisant de l'apprentissage par renforcement.

## Fichiers
Voici une description de l'organisation de ce projet :
```
MARLClassification
|
|-- data/
|     |-- mnist.py : Chargement des données MNIST
|
|-- environment/
|     |-- agent.py : Classe représentant un agent dans ce MARL
|     |-- core.py : Fonction de transition de l'environnement
|     |-- observation.py : Fonction donnant une obesrvation d'image MNIST selon une position
|     |-- transition.py : Fonction effectuant une transition selonune action
|
|-- nerworks/
|     |-- ft_extractor.py : NN pour l'extraction de features
|     |-- messages.py : NNs pour la reception et l'envoie de message entre agents
|     |-- models.py : Classe regroupant tout les NNs de ce package
|     |-- policy.py : NN pour la politique des agents
|     |-- prediction.py : NN pour la prédiction
|     |-- recurrents.py : NNs récurrents pour le Belief et Action
|
|-- res/
|     |-- downloaded/ : contient les données téléchargées (MNIST)
|     |-- img/ : contient les résultats du MARL
|     |-- download_mnist.sh : script pour télécharger les données MNIST
|
|-- main.py : Divers tests et fonction d'apprentissage sur MNIST
|-- tests.py : Test d'un CNN sur les données MNIST
```