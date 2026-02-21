"============================================================"
"Script Pharo - Export des metriques de qualite en CSV"
"Definit la classe MetricsExporter puis l utilise"
"============================================================"

"--- 1. Definir la classe MetricsExporter ---"
Object subclass: #MetricsExporter
    instanceVariableNames: 'model classes'
    classVariableNames: ''
    package: 'MGL843-Metrics'.

"--- 2. Compiler les methodes ---"

MetricsExporter compile: 'loadModelFrom: aFileReference
    "Charge un modele FamixTypeScript depuis un fichier JSON"
    model := FamixTypeScriptModel new.
    aFileReference readStreamDo: [ :readStream |
        model importFromJSONStream: readStream.
    ].
    model install.
    classes := model allWithType: FamixTypeScriptClass.
    Transcript show: ''Model loaded: '', classes size printString, '' classes found.''; cr.'
    classified: 'loading'.

MetricsExporter compile: 'computeMetricsForClass: aClass
    "Retourne un dictionnaire de metriques pour une classe donnee"
    | nbMethods nbAttrs loc |
    nbMethods := aClass methods size.
    nbAttrs := aClass attributes size.
    loc := aClass methods inject: 0 into: [ :sum :m |
        sum + ([ m numberOfLinesOfCode ifNil: [ 0 ] ] on: Error do: [ :e | 0 ])
    ].
    ^ Dictionary new
        at: ''name'' put: aClass name;
        at: ''methods'' put: nbMethods;
        at: ''attributes'' put: nbAttrs;
        at: ''loc'' put: loc;
        yourself.'
    classified: 'metrics'.

MetricsExporter compile: 'allMetrics
    "Retourne une collection de dictionnaires de metriques pour toutes les classes"
    ^ classes collect: [ :each | self computeMetricsForClass: each ].'
    classified: 'metrics'.

MetricsExporter compile: 'exportToCsv: aFilePath
    "Exporte les metriques au format CSV dans le fichier indique"
    | stream metrics |
    metrics := self allMetrics.
    stream := aFilePath asFileReference writeStream.
    stream nextPutAll: ''Nom_Classe;Nb_Methodes;Nb_Attributs;Lignes_de_Code''.
    stream cr.
    metrics do: [ :dict |
        | line |
        line := (dict at: ''name'') , '';'' ,
                (dict at: ''methods'') printString , '';'' ,
                (dict at: ''attributes'') printString , '';'' ,
                (dict at: ''loc'') printString.
        stream nextPutAll: line.
        stream cr.
        Transcript show: ''  '', (dict at: ''name''),
            '' | methods: '', (dict at: ''methods'') printString,
            '' | attrs: '', (dict at: ''attributes'') printString,
            '' | loc: '', (dict at: ''loc'') printString; cr.
    ].
    stream close.
    Transcript show: ''Export complete: '', aFilePath; cr.'
    classified: 'export'.

MetricsExporter compile: 'printSummary
    "Affiche un resume des metriques dans le Transcript"
    | metrics totalClasses totalMethods totalLoc |
    metrics := self allMetrics.
    totalClasses := metrics size.
    totalMethods := metrics inject: 0 into: [ :sum :d | sum + (d at: ''methods'') ].
    totalLoc := metrics inject: 0 into: [ :sum :d | sum + (d at: ''loc'') ].
    Transcript show: ''=== Summary ===''; cr.
    Transcript show: ''Total classes: '', totalClasses printString; cr.
    Transcript show: ''Total methods: '', totalMethods printString; cr.
    Transcript show: ''Total LOC: '', totalLoc printString; cr.'
    classified: 'printing'.

"============================================================"
"--- 3. Utiliser la classe MetricsExporter ---"
"============================================================"

| modelFile exporter |

"Localiser model.json"
modelFile := './model.json' asFileReference.
modelFile exists ifFalse: [
    | wsPath |
    wsPath := Smalltalk os environment at: 'GITHUB_WORKSPACE' ifAbsent: [ '.' ].
    modelFile := (wsPath , '/model.json') asFileReference.
].
modelFile exists ifFalse: [
    modelFile := FileLocator workingDirectory / 'model.json'.
].
modelFile exists ifFalse: [
    self error: 'model.json not found'.
].

Transcript show: '=== MGL843 Moose Metrics Export ==='; cr.
Transcript show: 'Model file: ', modelFile fullName; cr.

"Instancier et utiliser la classe"
exporter := MetricsExporter new.
exporter loadModelFrom: modelFile.
exporter exportToCsv: './export_metrics.csv'.
exporter printSummary.
