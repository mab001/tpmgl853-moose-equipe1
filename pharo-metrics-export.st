"============================================================"
"Script Pharo - Export des metriques de qualite en CSV"
"============================================================"

| modelFile model classes stream csvLine |

"--- 1. Localiser model.json ---"
modelFile := './model.json' asFileReference.
modelFile exists ifFalse: [
    | wsPath |
    wsPath := Smalltalk os environment at: 'GITHUB_WORKSPACE' ifAbsent: [ '.' ].
    modelFile := (wsPath , '/model.json') asFileReference.
].

Transcript show: '=== Moose Metrics Export ==='; cr.
Transcript show: 'Looking for: ', modelFile fullName; cr.
Transcript show: 'Exists: ', modelFile exists printString; cr.

modelFile exists ifFalse: [
    "Try current working directory"
    modelFile := FileLocator workingDirectory / 'model.json'.
    Transcript show: 'Trying: ', modelFile fullName; cr.
    Transcript show: 'Exists: ', modelFile exists printString; cr.
].

modelFile exists ifFalse: [
    Transcript show: 'ERROR: model.json not found'; cr.
    ^ self error: 'model.json not found'.
].

"--- 2. Charger le modele dans Moose ---"
Transcript show: 'Loading model...'; cr.

model := FamixTypeScriptModel new.
modelFile readStreamDo: [ :readStream |
    model importFromJSONStream: readStream.
].
model install.

Transcript show: 'Model loaded.'; cr.

"--- 3. Extraire les metriques ---"
classes := model allWithType: FamixTypeScriptClass.
Transcript show: 'Classes found: ', classes size printString; cr.

"Ecrire le CSV sans NeoCSVWriter (simple stream)"
stream := './export_metrics.csv' asFileReference writeStream.

"En-tete"
stream nextPutAll: 'Nom_Classe;Nb_Methodes;Nb_Attributs;Lignes_de_Code'.
stream cr.

"Parcourir chaque classe"
classes do: [ :each |
    | nom nbMethods nbAttrs loc |
    nom := each name.
    nbMethods := each methods size.
    nbAttrs := each attributes size.

    "Calculer LOC: sommer les LOC des methodes"
    loc := each methods inject: 0 into: [ :sum :m |
        sum + ([ m numberOfLinesOfCode ifNil: [ 0 ] ] on: Error do: [ :e | 0 ])
    ].

    Transcript show: '  ', nom,
        ' | methods: ', nbMethods printString,
        ' | attrs: ', nbAttrs printString,
        ' | loc: ', loc printString; cr.

    csvLine := nom , ';' , nbMethods printString , ';' , nbAttrs printString , ';' , loc printString.
    stream nextPutAll: csvLine.
    stream cr.
].

stream close.

Transcript show: 'Export complete: export_metrics.csv'; cr.
