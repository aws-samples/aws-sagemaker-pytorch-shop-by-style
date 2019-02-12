const gremlin = require('gremlin');
const Graph = gremlin.structure.Graph;
const DriverRemoteConnection = gremlin.driver.DriverRemoteConnection;
const order = gremlin.process.order;

const format_response = function(data){
    
    var payload = {"similarTo": []};

    for (var i in data) {
        
        var t = {
                    'brand': data[i].get('tprops').get('brand'),
                    'category': data[i].get('tprops').get('category'),
                    'subcategory': data[i].get('tprops').get('subcategory'),
                    'sku': data[i].get('tprops').get('product'),
                    'img': data[i].get('tpath'), 
                    'score': data[i].get('similarity')
                };
        
        payload.similarTo.push(t);
    }
    
   return {
       "statusCode": 200,
       "headers": {
           "Access-Control-Allow-Origin" : "*", // Required for CORS support to work
           "Access-Control-Allow-Credentials" : true // Required for cookies, authorization headers with HTTPS 
           },
        "body": JSON.stringify(payload),
        "isBase64Encoded": false
   };
};

exports.handler = (event, context, callback) => {

    const conn = process.env.DBCONN;
    const dc = new DriverRemoteConnection(conn);

    const graph = new Graph();
    const g = graph.traversal().withRemote(dc);
    
    const img = event["queryStringParameters"]["img"];
    var topN = parseInt(event["queryStringParameters"]["n"]);
    
    topN = Math.min(topN, 7);
        
    g.V()
    .hasLabel(img)
    .outE('similar').as('e')
    .inV().as('t').label().as('tpath')
    .select('e').values('similarity')
    .order().by(order.incr).limit(topN).as('similarity')
    .select('t').valueMap().as('tprops')
    .select('similarity','tpath', 'tprops')
    .toList().
        then(data => {
            console.log(data);
            callback(null, format_response(data));
            dc.close();
        }).catch(error => {
             callback(error,null);
            dc.close();       
        });
};