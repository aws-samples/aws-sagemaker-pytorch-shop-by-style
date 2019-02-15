## Shop-by-Style Experience using PyTorch and SageMaker

Contact: [Dylan Tong](mailto:dylatong@amazon.com)

The content in this repository was released as part of a [ML-blog post](https://amazon.awsapps.com/workdocs/index.html#/document/c557cbefa1f643b1ed3aade699052edbc3a1e6192e7c1a4fc24ede027105253f): A personalized 'shop-by-style' experience via PyTorch on Amazon SageMaker and Amazon Neptune

The following button will launch a CloudFormation template to deploy the shop-by-style prototype in us-west-2. There are no post-launch configuration steps required. The template will create IAM roles, network resources like a VPC, a CloudFront distribution, a Neptune database, API Gateway and Lambda resources. Thus, I recommend deploying this solution with admin permissions in a sandbox or personal account.

The prototype will work in other regions. However, it will require you to make modifications to the CloudFormation template, and copy the provided assets to the other region.

<a href="https://console.aws.amazon.com/cloudformation/home?region=us-west-
2#/stacks/new?stackName=shopbystyle-prototype&templateURL=https://s3-us-west-
2.amazonaws.com/reinvent2018-sagemaker-pytorch/cloudformation/blog/shop-by-
style/shopbystyle-prototype.yaml">
![launch stack button](/images/cloudformation-launch-stack.png)</a>

**Allow the template 45-60 minutes to launch. The prototype restores a Neptune database from a snapshot, which can take 45 minutes. As well, CloudFront will require some time before it can redirect traffic to the designated S3 origin. The reason is explained in the [docs](https://docs.aws.amazon.com/AmazonS3/latest/dev/VirtualHosting.html). If you experience an error loading the web pages, allow 20 minutes for DNS propagation to complete.**

Two URLs are provided in the Output section of the CloudFormation Template:

![CF Output](/images/cf_outputs.png)

ShoeRackPageURL links you to the webpage displayed below:

![Animated gif](/images/shopbystyle-ui-anim.gif)

GraphVisURL links you to a graph visualization sample page:

![Graph Viz](/images/graphvis.png)


## License

This library is licensed under the Apache 2.0 License. 
