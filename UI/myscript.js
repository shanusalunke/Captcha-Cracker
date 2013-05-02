//var file_check = 0; // variable for the timer functions
// Global Variables
var image_source="";
var algo="";
var top_n_count =0;
var top_count = 0;
var top_n_efficiency = 0;
var top_efficiency = 0;
var total_images=0;

// Functions
function testAlert(){
alert("this was a test");
}
function IsDocumentAvailable(url) {

        var fSuccess = false;
        var client = null;

        // XHR is supported by most browsers.
        // IE 9 supports it (maybe IE8 and earlier) off webserver
        // IE running pages off of disk disallows XHR unless security zones are set appropriately. Throws a security exception.
        // Workaround is to use old ActiveX control on IE (especially for older versions of IE that don't support XHR)

        // FireFox 4 supports XHR (and likely v3 as well) on web and from local disk

        // Works on Chrome, but Chrome doesn't seem to allow XHR from local disk. (Throws a security exception) No workaround known.

        try {
            client = new XMLHttpRequest();
            client.open("GET", url, false);
            client.send();
        }
        catch (err) {
            client = null;
        }

        // Try the ActiveX control if available
        if (client === null) {
            try {
                client = new ActiveXObject("Microsoft.XMLHTTP");
                client.open("GET", url, false);
                client.send();
            }
            catch (err) {
                // Giving up, nothing we can do
                client = null;
            }
        }

        fSuccess = Boolean(client && client.responseText);
        return fSuccess;
    }
	
function callDev(){
		
		var f = "C://Python27//Captcha_Cracker//cgi-bin//devcon.bat 'CPT 1'"
		//WshShell = new ActiveXObject( "WScript.Shell" );  
		//var r = WshShell.Run(f,1,false) ;  
		
		while(IsDocumentAvailable("myfile.html") !=true){
		}
		alert("file found");
		
	}

function DevCreateVectors(){
	var myform = document.forms["dev_corner_form"];
	var folderpath = myform["folder_path"].value ;
	var cpt_name =  myform["cpt_name"].value ;
	var set_type =  myform["set_Type"].value ;
	var folderpath_p = '\"' + myform["folder_path"].value + '\"';
	var cpt_name_p = '\"' + myform["cpt_name"].value +'\"';
	var set_type_p =  '\"' + set_type +'\"';
	//window.open("loading.hta", "_parent");
	var pyCall = "createVectors.py " + folderpath_p + " " + cpt_name_p+ " "+set_type;
	var f = 'C://Python27//Captcha_Cracker//cgi-bin//devcon.bat '+pyCall;
		WshShell = new ActiveXObject( "WScript.Shell" );  
		var r = WshShell.Run(f,1,false) ;  
	check_path = "C:\\Python27\\Captcha_Cracker\\Dataset Vectors\\"+cpt_name+".txt";
	//alert(check_path);
	//while(IsDocumentAvailable(check_path) !=true){
		// waiting till backend saves check_path
		//alert(check_path);
	
	//window.open("dev_corner.hta", "_parent");
	//document.getElementById('folder_path').value = folderpath;
	//document.getElementById('cpt_name').value = cpt_name;
	//alert(document.getElementById('folder_path').value);
}

function DevTrain(){
	// getting data from form
	var myform = document.forms["dev_corner_form"];
	var cpt_name = myform["cpt_name"].value;
	var cpt_name_p = '\"' + myform["cpt_name"].value +'\"';
	var set_type =  myform["set_Type"].value ;
	var set_type_p =  '\"' + set_type +'\"';
	// opening loading window
	//window.open("loading.hta", "_self");
	
	// preparing call
	var pyCall = "train.py " + cpt_name_p +" "+ set_type_p;
	var f = 'C://Python27//Captcha_Cracker//cgi-bin//devcon.bat '+pyCall;
		WshShell = new ActiveXObject( "WScript.Shell" );  
		var r = WshShell.Run(f,1,false) ;  
	check_path_1 = "C:\\Python27\\Captcha_Cracker\\Datasets\\"+cpt_name+".txt";
	check_path_2 = "C:\\Python27\\Captcha_Cracker\\Networks\\"+cpt_name+".txt";
	//alert(check_path);
	/*
	while(IsDocumentAvailable(check_path_1) !=true){
		// waiting till backend saves check_path
		//alert(check_path);
	}
	while(IsDocumentAvailable(check_path_2) !=true){
		// waiting till backend saves check_path
		//alert(check_path);
	}
	*/
	//file_check = window.setInterval(function(){IsDocumentAvailable(check_path_1)},1000);
	//file_check = window.setInterval(function(){IsDocumentAvailable(check_path_2)},1000);
	//window.open("dev_corner.hta", "_parent");
}

function DevRun(){
	var myform = document.forms["dev_corner_form"];
	var folderpath = myform["folder_path"].value ;
	var cpt_name =  myform["cpt_name"].value ;
	var set_type =  myform["set_Type"].value ;
}

function endsWith(str, suffix) {
    return str.indexOf(suffix, str.length - suffix.length) !== -1;
}

function getCracking(){
	var myform = document.forms["input_form"];
	//alert("getiing ");
	var folder_path = myform["folder_path"].value;
	var number = myform["number_crack"].value;
	if(!endsWith(folder_path,"\\")){
		folder_path = folder_path +"\\";
	}
	if (isNaN(number) || number==''){
		alert("Invalid Number of Images");
	}
	else{
	var folder_path_p = '\"' + folder_path +'\"';
	var number_p = '\"' + number +'\"';
	var algo_p = '\"' + algo +'\" ';
	// preparing call
	var pyCall = algo_p  + folder_path_p + " " +number_p;
	var f = 'C://Python27//Captcha_Cracker//cgi-bin//devcon.bat '+pyCall;
	WshShell = new ActiveXObject( "WScript.Shell" );  
	var r = WshShell.Run(f,1,false) ;  
	document.getElementById('view_result').style.visibility = 'visible';
	}
}

function setAlgo(i){
	algo = i
	//alert(algo);
}

function loadResult(){
// using ActiveX File System Object to load the temp.txt in Results

var fso = new ActiveXObject("Scripting.FileSystemObject"); 
var doc = "C:\\Python27\\Captcha_Cracker\\Results\\Temp\\temp.txt";
//alert("here");
while(! IsDocumentAvailable(doc)){
}
var s = fso.OpenTextFile(doc, 1, true);
var rows = s.ReadLine();
//alert(rows);
 rows = rows+ "<a class='button' href='javascript:createReport()'> Generate Report</a>" ;
 rows = rows+ "<a class='button' href='javascript:saveReport()'> Save Report</a>" ;
document.getElementById("mainframe").innerHTML = rows;
fso.Close
}

function createReport(){
// Creates a report based on the checked checkboxes
var top_n_check = document.getElementsByName("topn"),
top_n_count = 0;
for (var i=0; i<top_n_check.length; i++) {
    if (top_n_check[i].type === "checkbox" && top_n_check[i].checked === true) {
        top_n_count++;
    }
}

var top_check = document.getElementsByName("top"),
top_count = 0;
for (var i=0; i<top_check.length; i++) {
    if (top_check[i].type === "checkbox" && top_check[i].checked === true) {
        top_count++;
    }
}

total_images = i

top_n_efficiency = (top_n_count/total_images)*100;
top_efficiency = (top_count/total_images)*100;
message=""+
"Number of Images:: "+total_images+
"\nTop-N Efficiency:: "+top_n_efficiency+"%"+
"\nAll Correct Efficiency:: "+top_efficiency+"%";
alert(message);
}

function saveReport(){
// saves the report created above at a path specified by the user
var path = ""
path = prompt("Where do you want to save your file? ","C:\\Python27\\Captcha_Cracker\\Results\\myresult.txt");
while(path!=null && path!="" && IsDocumentAvailable(path)){
path = prompt("This path may be incorrect or already exist. Give the complete path (C:\\path\\to\\myfile.txt) ","C:\\Python27\\Captcha_Cracker\\Results\\myresult.txt");
}
var fso = new ActiveXObject("Scripting.FileSystemObject"),
f=fso.CreateTextFile(path,true);
report = "" +
"###################################################################\r\n"+
"#                       CAPTCHA CRACKER 1.0                       #\r\n"+
"#                              REPORT                             #\r\n"+
"#                                                                 #\r\n"+
"# DATE:"+Date()+"                                                 #\r\n"+
"# IMAGE SOURCE AT: "+image_source+"                               #\r\n"+
"# ALGORITHM USED: "+algo+"                                        #\r\n"+
"###################################################################\r\n"+
"\r\nNUMBER OF INPUT IMAGES:"+ total_images+
"\r\nTOP-N EFFICIENCY:"+top_n_efficiency+"%"+
"\r\nALL CHARACTERS CRACKED EFFICIENCY:"+top_efficiency+"%"+
"\r\n###################################################################";
f.WriteLine(report);
f.Close();
alert("Your file has been saved");
}

function showInfo(id){
	document.getElementById(id).style.display = 'inline';
}


function hideInfo(id){
	document.getElementById(id).style.display = 'none';
}
