// script to get the coordinates of the bounds of each visible layer in a photoshop file.
// The coordinates are reflected in left, top, right, bottom form, but can be modified to provide corner
//  coordinate pairs, or even fit a geojson type format
// Authored by Paul Riggot: https://community.adobe.com/t5/photoshop-ecosystem-discussions/export-co-ordinates-layer-bounds/td-p/4420832
// Modified by A.LaFevers

#target photoshop

main();

function main(){

if(!documents.length) return;

// the coords will be divided by 'scale', assumes the image is larger than the desired grid by a factor of 'scale'
var scale = 10;

var Info = getNamesPlusIDs().sort();

var Name = app.activeDocument.name.replace(/\.[^\.]+$/, '');

var path = app.activeDocument.path;

var file = new File(path + "/" + Name + "-coords.csv");

var h = parseInt(app.activeDocument.height.toString().replace(' px', ''));

file.open("w", "TEXT", "????");

$.os.search(/windows/i)  != -1 ? file.lineFeed = 'windows'  : file.lineFeed = 'macintosh';

file.writeln("zone,left,top,right,bottom");

for(var a in Info) {
	var l = Math.round(Info[a][1]/scale)
	var t = Math.round((Info[a][2]-h)/-scale)
	var r = Math.round(Info[a][5]/scale)
	var b = Math.round((Info[a][6]-h)/-scale)
	file.writeln(Info[a][0] +","+ l +","+ t + "," + r +","+ b);
}
file.close();

}

function getNamesPlusIDs(){

  var ref = new ActionReference();

  ref.putEnumerated( charIDToTypeID('Dcmn'), charIDToTypeID('Ordn'), charIDToTypeID('Trgt') );

  var count = executeActionGet(ref).getInteger(charIDToTypeID('NmbL')) +1;

  var Names=[];

try{

    activeDocument.backgroundLayer;

var i = 0; }catch(e){ var i = 1; };

  for(i;i<count;i++){

      if(i == 0) continue;

        ref = new ActionReference();

        ref.putIndex( charIDToTypeID( 'Lyr ' ), i );

        var desc = executeActionGet(ref);

        var layerName = desc.getString(charIDToTypeID( 'Nm  ' ));

        var Id = desc.getInteger(stringIDToTypeID( 'layerID' ));

        if(layerName.match(/^<\/Layer group/) ) continue;

        var vMask = desc.getBoolean(stringIDToTypeID('hasVectorMask' ));

    try{

      var adjust = typeIDToStringID(desc.getList (stringIDToTypeID('adjustment')).getClass (0));

      if(vMask == true){

          adjust = false;

          var Shape = true;

          }

      }catch(e){var adjust = false; var Shape = false;}

        var layerType = typeIDToStringID(desc.getEnumerationValue( stringIDToTypeID( 'layerSection' )));

        var isLayerSet =( layerType == 'layerSectionContent') ? false:true;

        var Vis = desc.getBoolean(stringIDToTypeID( 'visible' ));

        var descBounds = executeActionGet(ref).getObjectValue(stringIDToTypeID( "bounds" ));

        var X = descBounds.getUnitDoubleValue(stringIDToTypeID('left'));

        var Y = descBounds.getUnitDoubleValue(stringIDToTypeID('top'));
		
		var Rt = descBounds.getUnitDoubleValue(stringIDToTypeID('right'));

        var Bt = descBounds.getUnitDoubleValue(stringIDToTypeID('bottom'));

  var Wt = descBounds.getUnitDoubleValue(stringIDToTypeID('width'));

  var Ht = descBounds.getUnitDoubleValue(stringIDToTypeID('height'));

        if(Vis && !isLayerSet && !adjust) Names.push([[layerName], [X], [Y], [Wt], [Ht], [Rt], [Bt]]);

  };

return Names;

};