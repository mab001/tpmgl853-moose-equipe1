"============================================================"
"Script Pharo - Export des metriques de qualite en CSV"
"Definit la classe MetricsExporter puis l utilise"
"============================================================"
!

"--- 1. Definir la classe MetricsExporter ---"
Object subclass: #MetricsExporter
    instanceVariableNames: 'model classes'
    classVariableNames: ''
    package: 'MGL843-Metrics'.
!

"--- 2. Compiler les methodes ---"
MetricsExporter compile: 'loadModelFrom: aFileReference
    model := FamixTypeScriptModel new.
    aFileReference readStreamDo: [ :readStream |
        model importFromJSONStream: readStream.
    ].
    model install.
    classes := model allWithType: FamixTypeScriptClass.
    Transcript show: ''Model loaded: '', classes size printString, '' classes found.''; cr.'
    classified: 'loading'.
!

MetricsExporter compile: 'computeMetricsForClass: aClass
    | nbMethods nbAttrs loc ccTotal ccAvg locPerMethod |
    nbMethods := aClass methods size.
    nbAttrs := aClass attributes size.
    loc := aClass methods inject: 0 into: [ :sum :m |
        sum + ([ m numberOfLinesOfCode ifNil: [ 0 ] ] on: Error do: [ :e | 0 ])
    ].
    ccTotal := aClass methods inject: 0 into: [ :sum :m |
        sum + ([ m cyclomaticComplexity ifNil: [ 1 ] ] on: Error do: [ :e | 1 ])
    ].
    ccAvg := nbMethods > 0 ifTrue: [ ccTotal / nbMethods ] ifFalse: [ 0 ].
    locPerMethod := nbMethods > 0 ifTrue: [ loc / nbMethods ] ifFalse: [ 0 ].
    ^ Dictionary new
        at: ''name'' put: aClass name;
        at: ''methods'' put: nbMethods;
        at: ''attributes'' put: nbAttrs;
        at: ''loc'' put: loc;
        at: ''cc_total'' put: ccTotal;
        at: ''cc_avg'' put: ccAvg;
        at: ''loc_nom'' put: locPerMethod;
        yourself.'
    classified: 'metrics'.
!

MetricsExporter compile: 'allMetrics
    | filtered |
    filtered := classes select: [ :each | each name ~= ''Person'' ].
    ^ filtered collect: [ :each | self computeMetricsForClass: each ].'
    classified: 'metrics'.
!

MetricsExporter compile: 'exportToCsv: aFilePath
    | stream metrics |
    metrics := self allMetrics.
    stream := aFilePath asFileReference writeStream.
    stream nextPutAll: ''Nom_Classe;Nb_Methodes;Nb_Attributs;Lignes_de_Code;CC_Total;CC_Avg;LOC_NOM''.
    stream cr.
    metrics do: [ :dict |
        | line ccAvg locNom ccAvgStr locNomStr |
        ccAvg := [ dict at: ''cc_avg'' ] on: Error do: [ :e | 0 ].
        locNom := [ dict at: ''loc_nom'' ] on: Error do: [ :e | 0 ].
        ccAvgStr := (ccAvg asFloat round: 2) asString.
        locNomStr := (locNom asFloat round: 2) asString.
        line := (dict at: ''name'') , '';'' ,
                (dict at: ''methods'') printString , '';'' ,
                (dict at: ''attributes'') printString , '';'' ,
                (dict at: ''loc'') printString , '';'' ,
                (dict at: ''cc_total'') printString , '';'' ,
                ccAvgStr , '';'' ,
                locNomStr.
        stream nextPutAll: line.
        stream cr.
        Transcript show: ''  '', (dict at: ''name''),
            '' | methods: '', (dict at: ''methods'') printString,
            '' | loc: '', (dict at: ''loc'') printString,
            '' | CC_Total: '', (dict at: ''cc_total'') printString,
            '' | LOC/NOM: '', locNomStr; cr.
    ].
    stream close.
    Transcript show: ''Export complete: '', aFilePath; cr.'
    classified: 'export'.
!

MetricsExporter compile: 'printSummary
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
!

MetricsExporter compile: 'exportMethodsComplexityToCsv: aFilePath
    | stream allMethods |
    allMethods := model allWithType: FamixTypeScriptMethod.
    stream := aFilePath asFileReference writeStream.
    stream nextPutAll: ''ClassName;MethodName;Complexity;LOC''.
    stream cr.
    (allMethods sorted: [ :a :b | 
        ([ a cyclomaticComplexity ifNil: [ 1 ] ] on: Error do: [ :e | 1 ]) >
        ([ b cyclomaticComplexity ifNil: [ 1 ] ] on: Error do: [ :e | 1 ])
    ]) do: [ :method |
        | className cc loc |
        className := [ method parentType name ] on: Error do: [ :e | ''Unknown'' ].
        cc := [ method cyclomaticComplexity ifNil: [ 1 ] ] on: Error do: [ :e | 1 ].
        loc := [ method numberOfLinesOfCode ifNil: [ 0 ] ] on: Error do: [ :e | 0 ].
        stream nextPutAll: className , '';'' , method name , '';'' , cc printString , '';'' , loc printString.
        stream cr.
        Transcript show: ''  '', className, '' > '', method name, '' | CC: '', cc printString, '' | LOC: '', loc printString; cr.
    ].
    stream close.
    Transcript show: ''Method complexity export complete: '', aFilePath; cr.'
    classified: 'export'.
!

"============================================================"
"--- 3. Utiliser la classe MetricsExporter ---"
"============================================================"
| modelFile exporter wsPath |
"Now load the updated model"
modelFile := './tp3-notes-app.json' asFileReference.
modelFile exists ifFalse: [
    wsPath := Smalltalk os environment at: 'GITHUB_WORKSPACE' ifAbsent: [ '.' ].
    modelFile := (wsPath , '/tp3-notes-app.json') asFileReference.
].
modelFile exists ifFalse: [
    modelFile := FileLocator workingDirectory / 'tp3-notes-app.json'.
].
Transcript show: '=== MGL843 Moose Metrics Export ==='; cr.
Transcript show: 'Model file: ', modelFile fullName; cr.
exporter := MetricsExporter new.
exporter loadModelFrom: modelFile.
exporter exportToCsv: './export_metrics.csv'.
exporter exportMethodsComplexityToCsv: './export_methods_complexity.csv'.
exporter printSummary.
!
