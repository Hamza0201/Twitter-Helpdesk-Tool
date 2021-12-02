var database = firebase.database();

var clouddb = firebase.firestore();

var statusOptions = { "0" : "Open", "1" : "Processing", "2" : "Closed" };

var columnDefs = [
  {
	title: "T",
	targets : [0],
	type: "readonly",
	"width": "1%"
  },
{
	title: "Date",
	targets : [1],
	type: "readonly",
	"width": "7%"
  },
  {
	title: "Time",
	targets : [2],
	type: "readonly",
	"width": "5%"
  },
  {
	title: "User",
	targets : [3],
	type: "readonly",
	"visible": false
  },
 {
	title: "Tweet",
	targets : [4],
	type: "readonly",
  },
    {
	title: "Opinion",
	targets : [5],
	type: "readonly",
	"width": "7%"
  },
  {
	title: "View",
	targets : [6],
	type: "readonly",
	"data": null,
	"defaultContent": "<button type='button' class='btn btn-outline-info'>Link</button>",
	"width": "5%"
  },
 {
	title: "Status",
	targets : [7],
	orderData : [5, 0, 1],
	"width": "9%",
	type: "select",
	"options": [
	  "Open",
	  "Processing",
	  "Closed"
	]
  }];

var columnDefs2 = [{
	title: "T",
	type: "readonly",
	"width" : "1%"
  },
{
	title: "Date",
	type: "readonly",
	"width": "7%"
  },
  {
	title: "Time",
	type: "readonly",
	"width": "5%"
  },
  {
	title: "User",
	type: "readonly",
	"width" : "10%"
  },
 {
	title: "Tweet",
	type: "readonly",
  },
   {
	title: "Opinion",
	type: "readonly",
	"width" : "7%"
  },
  {
	title: "View",
	type: "readonly",
	"data": null,
	"width" : "7%",
	"defaultContent": "<button>Click!</button>"
  }
];

var columnDefs3 = [{
	title: "Email",
	targets : [0],
	type: "readonly",
	"width": "60%"
  },
  {
	title: "Account Created",
	targets : [1],
	type: "readonly",
	"width": "40%"
  }
];

var columnDefs4 = [{
	title: "Time",
	targets : [0],
	type: "readonly",
	"width": "30%"
  },
  {
	title: "Subject",
	targets : [1],
	type: "readonly",
	"width": "40%"
  },
  {
	title: "Description",
	targets : [2],
	type: "readonly",
	"width": "30%"
  }
];

var columnDefs5 = [{
	title: "Time",
	targets : [0],
	type: "readonly",
	"width": "15%"
  },
  {
	title: "Title",
	targets : [1],
	type: "readonly",
	"width": "20%"
  },
  {
	title: "Description",
	targets : [2],
	type: "readonly",
	"width": "65%"
  }
];

function doDate() {
	var dt = new Date();
	document.getElementById("datetime").innerHTML = dt.toLocaleString();
};

var getDaysArray = function(start, end) {
	for(var arr=[],dt=new Date(start); dt<=end; dt.setDate(dt.getDate()+1)){
		var dateObj = new Date(dt);
		var month = dateObj.getMonth() + 1; //months from 1-12
		var day = dateObj.getDate();
		var year = dateObj.getFullYear();
		if (month.toString().length < 2) month = '0' + month;
		if (day.toString().length < 2) day = '0' + day;
		newdate = day + "-" + month + "-" + year;
		arr.push(newdate);
	}
	return arr;
};

$(document).ready(function() {

	// Setup - add a text input to each footer cell
	$('#ex-table tfoot th').each( function () {
		var title = $(this).text();
		$(this).html( '<input type="text" class="form-control" placeholder="'+title+'" />' );
	} );

	var table = $('#ex-table').DataTable( {
		"order": [[ 2, "desc" ]],
		initComplete: function () {
			// Apply the search
			this.api().columns().every( function () {
				var that = this;
 
				$( 'input', this.footer() ).on( 'keyup change clear', function () {
					if ( that.search() !== this.value ) {
						that
							.search( this.value )
							.draw();
					}
				} );
			} );
		},
		scrollY: '55vh',
		scrollCollapse: true,
		pagin: false,
		columns: columnDefs,
		dom: '<"tickets-wrapper"B>frtip',
		select: 'single',
		responsive: true,
		altEditor: true,     // Enable altEditor
		buttons: [
			  {
				extend: 'selected', // Bind to Selected row
				text: 'Edit',
				name: 'edit'        // do not change name
			  }
		],
		onEditRow: function(datatable, rowdata, success, error) {
			var ref = database.ref().child('tweets');
			var key;
			ref.once("value")
			.then(function(snapshot) {
				snapshot.forEach(function(childSnapshot) {
					var total = childSnapshot.child("screen_name").val();
					if (rowdata[3] == total){
						key = childSnapshot.key;
						var ref2 = database.ref().child('tweets').child(String(key));
						ref2.update({ticket_status : rowdata[7]});
						location.reload();
					}
				});
			});
		}
	} );

	database.ref('tweets').on('child_added', function(snapshot){
		if(snapshot.exists()){
			var sentimence = null
			if (snapshot.val().sentimence < -0.05){
				sentimence = "<button type='button' class='btn btn-danger'>Negative</button>"
			}
			else if (snapshot.val().sentimence > 0.05){
				sentimence = "<button type='button' id='ticket-positive'class='btn btn-success'>Positive</button>"
			}
			else{
				sentimence = "<button type='button' class='btn btn-secondary'>Neutral</button>"
			}
			table.row.add( [
				snapshot.val().topic + 1,
				snapshot.val().date,
				snapshot.val().time,
				snapshot.val().screen_name,
				snapshot.val().processedticket,
				sentimence,
				snapshot.val().t_id,
				"<p id='ticket-status-text'>" + snapshot.val().ticket_status + "</p>"
			] ).draw();
		}
	});

	$('#ex-table tbody').on( 'click', 'button', function () {
		var data = table.row( $(this).parents('tr') ).data();
		let twitterlink = "https://twitter.com/";
		twitterlink += data[3];
		twitterlink += "/status/";
		twitterlink += data[6];
		window.open(twitterlink);
	} );

//THIS IS THE Archive reports TABLE

// This is the Admin table
	var admin_table = $('#admin-table').DataTable( {
		scrollY: '60vh',
		scrollCollapse: true,
		pagin: false,
		columns: columnDefs3,
		select: 'single',
		responsive: true
	} );

	database.ref('users').on('child_added', function(snapshot){
		if(snapshot.exists()){
			admin_table.row.add( [
				snapshot.val().email,
				snapshot.val().createdAt
			] ).draw();
		}
	});

	$('#adminmaster').on('click', function () {
		let twitterlink = "https://console.firebase.google.com/u/1/project/twitter-d34a5/authentication/users";
		window.open(twitterlink);
	} );

	setInterval(doDate,1000);

	$(function () {
		$('[data-toggle="tooltip"]').tooltip()
	})

	$("#search-kb").on('change paste input', function(){
		var context = document.querySelectorAll("p.mb-1"); // requires an element with class "context" to exist
		var instance1 = new Mark(context);
		instance1.unmark();
		var instance = new Mark(document.querySelectorAll("p.mb-1"));
      	instance.mark(document.getElementById("search-kb").value, {
    	"element": "span",
    	"className": "highlight"
		});
	});

	$('.dashboard-wrapper').on('click', function () {
		var context = document.querySelectorAll("p.mb-1"); // requires an element with class "context" to exist
		var instance = new Mark(context);
		instance.unmark();
	} );

	$('#ticket-table tfoot th').each( function () {
		var title = $(this).text();
		$(this).html( '<input type="text" class="form-control" placeholder="'+title+'" />' );
	} );

//This si the ticket report support table
	var ticket_table = $('#ticket-table').DataTable( {
		scrollY: '60vh',
		scrollCollapse: true,
		pagin: false,
		columns: columnDefs4,
		select: 'single',
		responsive: true
	} );

	$('#ticket-table tbody').on( 'click', 'tr', function () {
        if ( $(this).hasClass('selected') ) {
            $(this).removeClass('selected');
        }
        else {
            ticket_table.$('tr.selected').removeClass('selected');
            $(this).addClass('selected');
        }
    } );
 
    $('#ticket-delete').click( function () {
    	var rowdata = ticket_table.row('.selected').data();
		var ref = database.ref().child('tickets');
		ref.once("value")
		.then(function(snapshot) {
			snapshot.forEach(function(childSnapshot) {
				var title = childSnapshot.child("title").val();
				var description = childSnapshot.child("description").val();
				if ((rowdata[1] == title) && (rowdata[2] == description)){
					ref.child(childSnapshot.key).remove();
					ticket_table.row('.selected').remove().draw( false );
				}
			});
		});
    } );

	database.ref('tickets').on('child_added', function(snapshot){
		if(snapshot.exists()){
			ticket_table.row.add( [
				snapshot.val().time,
				snapshot.val().title,
				snapshot.val().description
			] ).draw();
		}
	});

    $('.dropdown-menu a').click(function () {           
    	$('#ngrambutton').text($(this).text());
    	if ($(this).text() == 'N = 1'){
    		document.getElementById("2-gram").style.display= "none";
    		document.getElementById("3-gram").style.display= "none";
    		document.getElementById("1-gram").style.display= "inline-block";
    	}
    	else if ($(this).text() == 'N = 2'){
    		document.getElementById("1-gram").style.display= "none";
    		document.getElementById("3-gram").style.display= "none";
    		document.getElementById("2-gram").style.display= "inline-block";
    	}
    	else {
    		document.getElementById("1-gram").style.display= "none";
    		document.getElementById("2-gram").style.display= "none";
    		document.getElementById("3-gram").style.display= "inline-block";
    	}
  	});

} );


function getData() {
		var dateObj = new Date();
		var month = dateObj.getMonth() + 1; //months from 1-12
		var day = dateObj.getDate();
		var year = dateObj.getFullYear();
		if (month.toString().length < 2) month = '0' + month;
		if (day.toString().length < 2) day = '0' + day;
		newdate = day + "-" + month + "-" + year;
		var time = dateObj.getHours() + ":" + dateObj.getMinutes() + ":" + dateObj.getSeconds();
		var milliseconds = Date.parse(dateObj);
		milliseconds = milliseconds - (1.5 * 60 * 1000)
		d = new Date(milliseconds)
		var hours = d.getHours();
		var minutes = d.getMinutes();
		var seconds = d.getSeconds();
		if (hours.toString().length < 2) hours = '0' + hours;
		if (minutes.toString().length < 2) minutes = '0' + minutes;
		if (seconds.toString().length < 2) seconds = '0' + seconds;
		var old_time = hours + ":" + minutes + ":" + seconds;
		let counter = 0;
		database.ref().child('counter').set({'counter': 0}).then().catch();
        database.ref().child('tweets').once("value")
		.then(function(snapshot) {
			snapshot.forEach(function(childSnapshot) {
				var date = childSnapshot.child("date").val();
				var new_time = childSnapshot.child("time").val();
				if ((date == newdate) && (new_time >= old_time)) {
					counter++;
				}
			});
			database.ref().child('counter').set({'counter': counter}).then().catch();
		});
	}

function getLiveData() {
		var dateObj = new Date();
		var month = dateObj.getMonth() + 1; //months from 1-12
		var day = dateObj.getDate();
		var year = dateObj.getFullYear();
		if (month.toString().length < 2) month = '0' + month;
		if (day.toString().length < 2) day = '0' + day;
		newdate = day + "-" + month + "-" + year;
		var time = dateObj.getHours() + ":" + dateObj.getMinutes() + ":" + dateObj.getSeconds();
		var milliseconds = Date.parse(dateObj);
		milliseconds = milliseconds - (1.5 * 60 * 1000)
		d = new Date(milliseconds)
		var hours = d.getHours();
		var minutes = d.getMinutes();
		var seconds = d.getSeconds();
		if (hours.toString().length < 2) hours = '0' + hours;
		if (minutes.toString().length < 2) minutes = '0' + minutes;
		if (seconds.toString().length < 2) seconds = '0' + seconds;
		var old_time = hours + ":" + minutes + ":" + seconds;
		let positive = 0;
		let neutral = 0;
		let negative = 0;
		database.ref().child('counter').set({'positive': 0, 'neutral' : 0, 'negative' : 0}).then().catch();
        database.ref().child('tweets').once("value")
		.then(function(snapshot) {
			snapshot.forEach(function(childSnapshot) {
				var date = childSnapshot.child("date").val();
				var new_time = childSnapshot.child("time").val();
				if ((date == newdate) && (new_time >= old_time)) {
					if (childSnapshot.child('sentimence').val() < -0.05){
						negative++;
					}
					else if ((childSnapshot.child('sentimence').val() >= -0.05) & (childSnapshot.child('sentimence').val() <= 0.05)){
						positive++;
					}
					else{
						neutral++;
					}
				}
			});
			database.ref().child('counter').set({'positive': positive}).then().catch();
			database.ref().child('counter').set({'neutral': neutral}).then().catch();
			database.ref().child('counter').set({'negative': negative}).then().catch();
		});
	}

var windowLoc = $(location).attr('pathname');
switch(windowLoc){      
  case "/home":
    var config = {responsive: true, displayModeBar: false}

	var layout = {
		autosize: false,
		width: 820,
		height: 280,
		margin: {
		    l: 20,
		    r: 30,
		    b: 30,
		    t: 10,
		    pad: 1
		},
		font: {size: 12},
		    xaxis: {
		    autotick: false,
		    ticks: 'outside'
		},
		yaxis: {
		    autotick: false,
		    ticks: 'inside',
		    tick0: 0,
		    dtick: 5
		}
	};

	Plotly.plot('chart-dashboard',[{
    	x:[undefined],
        y:[0],
        type:'scatter',
        fill:"tozeroy",
        fillcolor: 'rgba(221, 221, 199, 0.5)',
        name:'Open',
        line: {
        	color: 'rgb(221, 221, 119)'
        }
    },
    {
    	x:[undefined],
        y:[0],
        type:'scatter',
        fill:"tozeroy",
        fillcolor: 'rgba(112, 212, 180, 0.5)',
        name:'Processing',
        line: {
        	color: 'rgb(112, 212, 180)'
        }
    },
    {
    	x:[undefined],
        y:[0],
        type:'scatter',
        fill:"tozeroy",
        fillcolor: 'rgba(61, 6, 90, 0.5)',
        name:'Closed',
        line: {
        	color: 'rgb(61, 6, 90)'
        }
    }], layout, config);

    database.ref().child('ticket_recorder').once("value")
	.then(function(snapshot) {
		snapshot.forEach(function(childSnapshot) {
			var time = childSnapshot.child("time").val();
			var new_time = time.substring(0,5);
			var open_tickets = childSnapshot.child("open_tickets").val();
			var processing_tickets = childSnapshot.child("processing_tickets").val();
			var closed_tickets = childSnapshot.child("closed_tickets").val();
			Plotly.extendTraces('chart-dashboard',{x:[[new_time] ,[new_time], [new_time]], y:[[open_tickets], [processing_tickets], [closed_tickets]]},[0, 1 ,2]);
		});
	});
    
    var cnt = 0;
    setInterval(function(){
		database.ref().child("live_counter").get().then(function(snapshot) {
		var time = snapshot.child("time").val();
		var new_time = time.substring(0,5);
		var open_tickets = snapshot.val().open_tickets;
		var processing_tickets = snapshot.val().processing_tickets;
		var closed_tickets = snapshot.val().closed_tickets;
        Plotly.extendTraces('chart-dashboard',{x:[[new_time] ,[new_time], [new_time]], y:[[open_tickets], [processing_tickets], [closed_tickets]]},[0, 1 ,2]);
            });
        cnt++;
        if(cnt > 500) {
            Plotly.relayout('chart-dashboard',{
                xaxis: {
                    range: [cnt-500,cnt]
                }
            });
        }
    },300000);
    break;
    case "/notes":
        var hidden_email = document.getElementById('session-email').innerHTML;

    //This si the Notes table
	var note_table = $('#note-table').DataTable( {
		scrollY: '60vh',
		scrollCollapse: true,
		pagin: false,
		columns: columnDefs5,
		select: 'single',
		dom: 'rtp',
		responsive: true
	} );

	$('#note-table tbody').on( 'click', 'tr', function () {
        if ( $(this).hasClass('selected') ) {
            $(this).removeClass('selected');
        }
        else {
            note_table.$('tr.selected').removeClass('selected');
            $(this).addClass('selected'); 
        }
    } );
 
    $('#note-delete').click( function () {
    	var rowdata = note_table.row('.selected').data();
		database.ref().child('users').once("value")
		.then(function(snapshot) {
			snapshot.forEach(function(childSnapshot) {
				var email = childSnapshot.child("email").val();
				if (email == hidden_email){
					database.ref().child('users/' + childSnapshot.key + '/notes').once("value")
					.then(function(snapshot) {
						snapshot.forEach(function(childChildSnapshot) {
							var title = childChildSnapshot.child("title").val();
							var description = childChildSnapshot.child("description").val();
							if ((rowdata[1] == title) && (rowdata[2] == description)){
								database.ref().child('users').child(childSnapshot.key).child('notes').child(childChildSnapshot.key).remove();
								note_table.row('.selected').remove().draw( false );
							}
						});
					});
				}
			});
		});
    } );

    database.ref().child('users').once("value")
		.then(function(snapshot) {
			snapshot.forEach(function(childSnapshot) {
				var email = childSnapshot.child("email").val();
				if (email == hidden_email){
					database.ref('users/' + childSnapshot.key + '/notes').on('child_added', function(snapshot){
						if(snapshot.exists()){
							note_table.row.add( [
								snapshot.val().time,
								snapshot.val().title,
								snapshot.val().description
							] ).draw();
						}
					});
				}
			});
		});
	break;
	case "/trends":
		var loc = window.location.pathname;
		var dir = loc.substring(0, loc.lastIndexOf('/'));
		$('#ml').on('click', function () {
		let twitterlink = dir + '/lda';
		window.open(twitterlink);
		} );
	    var config = {responsive: true, displayModeBar: false}

		var layout = {
			autosize: false,
			width: 1100,
		  	height: 280,
		  	margin: {
		    l: 25,
		    r: -5,
		    b: 50,
		    t: 0,
		    pad: 2
		  },
		  font: {size: 12},
		    xaxis: {
		    autotick: false,
		    ticks: 'outside'
		  },
		  yaxis: {
		    autotick: false,
		    ticks: 'outside',
		    tick0: 0,
		    dtick: 5
		  }
			};

		Plotly.plot('chart',[{
	    	x:[undefined],
	        y:[0],
	        type:'scatter',
	        fill:"tozeroy",
	        fillcolor: 'rgba(41, 112, 184, 0.5)',
	        name:'Neutral',
	        line: {
	        	color: 'rgb(41, 112, 184)'
	        }
	    },
	    {
	    	x:[undefined],
	        y:[0],
	        type:'scatter',
	        fill:"tozeroy",
	        fillcolor: 'rgba(26, 35, 126, 0.6)',
	        name:'Negative',
	        line: {
	        	color: 'rgb(26, 35, 126)'
	        }
	    },
	    {
	    	x:[undefined],
	        y:[0],
	        type:'scatter',
	        fill:"tozeroy",
	        fillcolor: 'rgba(78, 188, 213, 0.5)',
	        name:'Positive',
	        line: {
	        	color: 'rgb(78, 188, 213)'
	        }
	    }], layout, config);

		database.ref().child('sentiment_recorder').once("value")
		.then(function(snapshot) {
			snapshot.forEach(function(childSnapshot) {
				var time = childSnapshot.child("time").val();
				var new_time = time.substring(0,5);
				var positive = childSnapshot.child("positive").val();
				var negative = childSnapshot.child("negative").val();
				var neutral = childSnapshot.child("neutral").val();
				Plotly.extendTraces('chart',{x:[[new_time] ,[new_time], [new_time]], y:[[neutral], [-negative], [positive]]},[0, 1 ,2]);
			});
		});
	    
	    var cnt = 0;
	    setInterval(function(){
			database.ref().child("live_counter").get().then(function(snapshot) {
			var dateObj = new Date();
	    	var milliseconds = Date.parse(dateObj);
			milliseconds = milliseconds - (0.01 * 60 * 1000)
			d = new Date(milliseconds)
			var hours = d.getHours();
			var minutes = d.getMinutes();
			var seconds = d.getSeconds();
			if (hours.toString().length < 2) hours = '0' + hours;
			if (minutes.toString().length < 2) minutes = '0' + minutes;
			if (seconds.toString().length < 2) seconds = '0' + seconds;
			var old_time = hours + ":" + minutes;
			var negative = snapshot.val().negative;
			var neutral = snapshot.val().neutral;
			var positive = snapshot.val().positive;
			var count = negative + neutral + positive;
	        Plotly.extendTraces('chart',{x:[[old_time] ,[old_time], [old_time]], y:[[neutral], [-negative], [positive]]},[0, 1 ,2]);
	            });
	        cnt++;
	        if(cnt > 500) {
	            Plotly.relayout('chart',{
	                xaxis: {
	                    range: [cnt-500,cnt]
	                }
	            });
	        }
	    },300000);
	    break;
	case "/reports":
		$('#reports-table tfoot th').each( function () {
		var title = $(this).text();
		$(this).html( '<input type="text" class="form-control" placeholder="'+title+'" />' );
		} );

		var reports_table = $('#reports-table').DataTable( {
			initComplete: function () {
				// Apply the search
				this.api().columns().every( function () {
					var that = this;
	 
					$( 'input', this.footer() ).on( 'keyup change clear', function () {
						if ( that.search() !== this.value ) {
							that
								.search( this.value )
								.draw();
						}
					} );
				} );
			},
			scrollY: '60vh',
			scrollCollapse: true,
			pagin: false,
			columns: columnDefs2,
			select: 'single',
			responsive: true,
			dom: '<"reports-wrapper"B>frtip',
			buttons: [
				'copy', 'csv', 'excel', 'pdf', 'print'
			]
		} );

		var start = moment().subtract(29, 'days');
		var end = moment();

		function cb(start, end) {
			$('#reportrange span').html(start.format('MMMM D, YYYY') + ' - ' + end.format('MMMM D, YYYY'));
			reports_table.clear().draw();
			var daylist = getDaysArray(new Date(start.format('YYYY-MM-DD')),new Date(end.format('YYYY-MM-DD')));
			daylist.forEach(function(entry) {
				clouddb.collection(entry).get().then((querySnapshot) => {
					querySnapshot.forEach((doc) => {
						var sentimence = null;
						if (doc.data().sentimence < -0.05){
							sentimence = "<button type='button' class='btn btn-danger'>Negative</button>"
						}
						else if (doc.data().sentimence > 0.05){
							sentimence = "<button type='button' id='ticket-positive'class='btn btn-success'>Positive</button>"
						}
						else{
							sentimence = "<button type='button' class='btn btn-secondary'>Neutral</button>"
						}
						reports_table.row.add( [
							doc.data().topic + 1,
							doc.data().date,
							doc.data().time,
							doc.data().screen_name,
							doc.data().processedticket,
							sentimence,
							doc.data().t_id
						] ).draw();
					});
				})
				.catch((error) => {
					console.log("Error getting documents: ", error);
				});
			});

		}

		$('#reportrange').daterangepicker({
			startDate: start,
			endDate: end,
			ranges: {
			   'Today': [moment(), moment()],
			   'Yesterday': [moment().subtract(1, 'days'), moment().subtract(1, 'days')],
			   'Last 7 Days': [moment().subtract(6, 'days'), moment()],
			   'Last 30 Days': [moment().subtract(29, 'days'), moment()],
			   'This Month': [moment().startOf('month'), moment().endOf('month')],
			   'Last Month': [moment().subtract(1, 'month').startOf('month'), moment().subtract(1, 'month').endOf('month')]
			}
		}, cb);

		cb(start, end);

		$('#reports-table tbody').on( 'click', 'button', function () {
			var data = reports_table.row( $(this).parents('tr') ).data();
			let twitterlink = "https://twitter.com/";
			twitterlink += data[3];
			twitterlink += "/status/";
			twitterlink += data[6];
			window.open(twitterlink);
		} );
		break;
}
