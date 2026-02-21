"============================================================"
"Script Pharo - Export des métriques de qualité en CSV"
"============================================================"
"Ce script est exécuté dans une image Moose par smalltalkCI."
"Il effectue 3 étapes :"
"  1. Charger le modèle Famix (model.json) généré par ts2famix"
"  2. Extraire les métriques de chaque classe (NOM, NOA, LOC)"
"  3. Écrire les résultats dans export_metrics.csv"
"============================================================"

| modelFile model classes stream writer |

"--- 1. Localiser le fichier model.json ---"
"On cherche d'abord dans le répertoire courant,"
"puis dans GITHUB_WORKSPACE si on est en CI."
modelFile := './model.json' asFileReference.

modelFile exists ifFalse: [
    | wsPath |
    wsPath := (Smalltalk os environment at: 'GITHUB_WORKSPACE' ifAbsent: [ '.' ]).
    modelFile := (wsPath , '/model.json') asFileReference.
].

Transcript show: '=== Moose Metrics Export ==='; cr.
Transcript show: 'Model file: ', modelFile fullName; cr.
Transcript show: 'Exists: ', modelFile exists printString; cr.

modelFile exists ifFalse: [
    Transcript show: '❌ model.json introuvable!'; cr.
    self error: 'model.json not found at: ', modelFile fullName.
].

"--- 2. Charger le modèle dans Moose ---"
Transcript show: 'Chargement du modèle...'; cr.

modelFile readStreamDo: [ :readStream |
    model := FamixTypeScriptModel new importFromJSONStream: readStream.
    model install.
].

Transcript show: 'Modèle chargé avec succès.'; cr.

"--- 3. Extraire les métriques et écrire le CSV ---"
classes := model allModelClasses.
Transcript show: 'Nombre de classes trouvées: ', classes size printString; cr.

"Créer le fichier CSV avec séparateur ;"
stream := './export_metrics.csv' asFileReference writeStream.
writer := NeoCSVWriter on: stream.
writer separator: $;.

"En-tête"
writer nextPut: #( 'Nom_Classe' 'Nb_Methodes' 'Nb_Attributs' 'Lignes_de_Code' ).

"Parcourir chaque classe"
classes do: [ :each |
    | loc anchor srcFile content snippet |
    loc := 0.

    "Essayer de calculer LOC depuis le fichier source"
    anchor := each sourceAnchor.
    (anchor notNil and: [ anchor isKindOf: FamixTypeScriptIndexedFileAnchor ]) ifTrue: [
        [ 
            srcFile := anchor fileName asFileReference.
            srcFile exists ifTrue: [
                content := srcFile contents.
                snippet := content copyFrom: anchor startPos to: (anchor endPos min: content size).
                loc := snippet lineCount.
            ].
        ] on: Error do: [ :e |
            Transcript show: '  ⚠ LOC error for ', each name, ': ', e messageText; cr.
        ].
    ].

    "Fallback : sommer les LOC des méthodes"
    loc = 0 ifTrue: [
        loc := each methods inject: 0 into: [ :sum :m | 
            sum + (m numberOfLinesOfCode ifNil: [ 0 ]) 
        ].
    ].

    Transcript show: '  ', each name, 
        ' | methods:', each numberOfMethods printString,
        ' | attrs:', each numberOfAttributes printString,
        ' | loc:', loc printString; cr.

    writer nextPut: {
        each name.
        each numberOfMethods.
        each numberOfAttributes.
        loc.
    }.
].

stream close.

Transcript show: '✅ Export terminé: export_metrics.csv'; cr.
