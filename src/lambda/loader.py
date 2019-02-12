import json
import urllib.request
import urllib.parse

def lambda_handler(event, context):
    
    #loaderEndpoint = event.ResourceProperties.LoaderEndpoint
    #verticesDataLocation = event.ResourceProperties.VerticesDataLocation
    #edgesDataLocation = event.ResourceProperties.EdgesDataLocation
    #iamRoleArn = event.ResourceProperties.IAMRoleARN
    #region = event.ResourceProperties.Region
    
    loaderEndpoint = event['ResourceProperties']['LoaderEndpoint']
    verticesDataLocation = event['ResourceProperties']['VerticesDataLocation']
    dataLocation = event['ResourceProperties']['DataLocation']
    edgesDataLocation = event['ResourceProperties']['EdgesDataLocation']
    iamRoleArn = event['ResourceProperties']['IAMRoleARN']
    region = event['ResourceProperties']['Region']
    
    url = loaderEndpoint

    headers = {"Content-Type": "application/json"}
    
    params = {"source":dataLocation,
            "format":"csv",
            "iamRoleArn":iamRoleArn,
            "region":region}

    data = json.dumps(params).encode('utf-8')
    
    status = 200
    
    try:
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req) as response:
            rdata =json.loads(response.read().decode('utf-8'))
            print(rdata)

    except:
        print("POST failed.")
        status = 500
        
    return {
        "status": status,
        "body": rdata["payload"]
    }