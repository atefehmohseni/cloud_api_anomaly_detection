API_PARAMETERS_FILE = "aws_apis_json.txt"


def print_red(skk): print("\033[91m {}\033[00m" .format(skk))


def print_green(skk): print("\033[92m {}\033[00m" .format(skk))


def print_yellow(skk): print("\033[93m {}\033[00m" .format(skk))

service_dict = {'accessanalyzer.amazonaws.com': 1,
                'acm.amazonaws.com': 2,
                'acm-pca.amazonaws.com': 3,
                'alexaforbusiness.amazonaws.com': 4,
                'amplify.amazonaws.com': 5,
                'apigateway.amazonaws.com': 6,
                'apigatewaymanagementapi.amazonaws.com': 7,
                'apigatewayv2.amazonaws.com': 8,
                'appconfig.amazonaws.com': 9,
                'application-autoscaling.amazonaws.com': 10,
                'application-insights.amazonaws.com': 11,
                'appmesh.amazonaws.com': 12,
                'appstream.amazonaws.com': 13,
                'appsync.amazonaws.com': 14,
                'athena.amazonaws.com': 15,
                'autoscaling.amazonaws.com': 16,
                'autoscaling-plans.amazonaws.com': 17,
                'backup.amazonaws.com': 18,
                'batch.amazonaws.com': 19,
                'budgets.amazonaws.com': 20,
                'ce.amazonaws.com': 21,
                'chime.amazonaws.com': 22, 'cloud9.amazonaws.com': 23, 'clouddirectory.amazonaws.com': 24, 'cloudformation.amazonaws.com': 25, 'cloudfront.amazonaws.com': 26, 'cloudhsm.amazonaws.com': 27, 'cloudhsmv2.amazonaws.com': 28, 'cloudsearch.amazonaws.com': 29, 'cloudsearchdomain.amazonaws.com': 30, 'cloudtrail.amazonaws.com': 31, 'cloudwatch.amazonaws.com': 32, 'codebuild.amazonaws.com': 33, 'codecommit.amazonaws.com': 34, 'codedeploy.amazonaws.com': 35, 'codeguru-reviewer.amazonaws.com': 36, 'codeguruprofiler.amazonaws.com': 37, 'codepipeline.amazonaws.com': 38, 'codestar.amazonaws.com': 39, 'codestar-connections.amazonaws.com': 40, 'codestar-notifications.amazonaws.com': 41, 'cognito-identity.amazonaws.com': 42, 'cognito-idp.amazonaws.com': 43, 'cognito-sync.amazonaws.com': 44, 'comprehend.amazonaws.com': 45, 'comprehendmedical.amazonaws.com': 46, 'compute-optimizer.amazonaws.com': 47, 'config.amazonaws.com': 48, 'connect.amazonaws.com': 49, 'connectparticipant.amazonaws.com': 50, 'cur.amazonaws.com': 51, 'dataexchange.amazonaws.com': 52, 'datapipeline.amazonaws.com': 53, 'datasync.amazonaws.com': 54, 'dax.amazonaws.com': 55, 'detective.amazonaws.com': 56, 'devicefarm.amazonaws.com': 57, 'directconnect.amazonaws.com': 58, 'discovery.amazonaws.com': 59, 'dlm.amazonaws.com': 60, 'dms.amazonaws.com': 61, 'docdb.amazonaws.com': 62, 'ds.amazonaws.com': 63, 'dynamodb.amazonaws.com': 64, 'dynamodbstreams.amazonaws.com': 65, 'ebs.amazonaws.com': 66, 'ec2.amazonaws.com': 67, 'ec2-instance-connect.amazonaws.com': 68, 'ecr.amazonaws.com': 69, 'ecs.amazonaws.com': 70, 'efs.amazonaws.com': 71, 'eks.amazonaws.com': 72, 'elastic-inference.amazonaws.com': 73, 'elasticache.amazonaws.com': 74, 'elasticbeanstalk.amazonaws.com': 75, 'elastictranscoder.amazonaws.com': 76, 'elb.amazonaws.com': 77, 'elbv2.amazonaws.com': 78, 'emr.amazonaws.com': 79, 'es.amazonaws.com': 80, 'events.amazonaws.com': 81, 'firehose.amazonaws.com': 82, 'fms.amazonaws.com': 83, 'forecast.amazonaws.com': 84, 'forecastquery.amazonaws.com': 85, 'frauddetector.amazonaws.com': 86, 'fsx.amazonaws.com': 87, 'gamelift.amazonaws.com': 88, 'glacier.amazonaws.com': 89, 'globalaccelerator.amazonaws.com': 90, 'glue.amazonaws.com': 91, 'greengrass.amazonaws.com': 92, 'groundstation.amazonaws.com': 93, 'guardduty.amazonaws.com': 94, 'health.amazonaws.com': 95, 'iam.amazonaws.com': 96, 'imagebuilder.amazonaws.com': 97, 'importexport.amazonaws.com': 98, 'inspector.amazonaws.com': 99, 'iot.amazonaws.com': 100, 'iot-data.amazonaws.com': 101, 'iot-jobs-data.amazonaws.com': 102, 'iot1click-devices.amazonaws.com': 103, 'iot1click-projects.amazonaws.com': 104, 'iotanalytics.amazonaws.com': 105, 'iotevents.amazonaws.com': 106, 'iotevents-data.amazonaws.com': 107, 'iotsecuretunneling.amazonaws.com': 108, 'iotsitewise.amazonaws.com': 109, 'iotthingsgraph.amazonaws.com': 110, 'kafka.amazonaws.com': 111, 'kendra.amazonaws.com': 112, 'kinesis.amazonaws.com': 113, 'kinesis-video-archived-media.amazonaws.com': 114, 'kinesis-video-media.amazonaws.com': 115, 'kinesis-video-signaling.amazonaws.com': 116, 'kinesisanalytics.amazonaws.com': 117, 'kinesisanalyticsv2.amazonaws.com': 118, 'kinesisvideo.amazonaws.com': 119, 'kms.amazonaws.com': 120, 'lakeformation.amazonaws.com': 121, 'lambda.amazonaws.com': 122, 'lex-models.amazonaws.com': 123, 'lex-runtime.amazonaws.com': 124, 'license-manager.amazonaws.com': 125, 'lightsail.amazonaws.com': 126, 'logs.amazonaws.com': 127, 'machinelearning.amazonaws.com': 128, 'macie.amazonaws.com': 129, 'managedblockchain.amazonaws.com': 130, 'marketplace-catalog.amazonaws.com': 131, 'marketplace-entitlement.amazonaws.com': 132, 'marketplacecommerceanalytics.amazonaws.com': 133, 'mediaconnect.amazonaws.com': 134, 'mediaconvert.amazonaws.com': 135, 'medialive.amazonaws.com': 136, 'mediapackage.amazonaws.com': 137, 'mediapackage-vod.amazonaws.com': 138, 'mediastore.amazonaws.com': 139, 'mediastore-data.amazonaws.com': 140, 'mediatailor.amazonaws.com': 141, 'meteringmarketplace.amazonaws.com': 142, 'mgh.amazonaws.com': 143, 'migrationhub-config.amazonaws.com': 144, 'mobile.amazonaws.com': 145, 'mq.amazonaws.com': 146, 'mturk.amazonaws.com': 147, 'neptune.amazonaws.com': 148, 'networkmanager.amazonaws.com': 149, 'opsworks.amazonaws.com': 150, 'opsworkscm.amazonaws.com': 151, 'organizations.amazonaws.com': 152, 'outposts.amazonaws.com': 153, 'personalize.amazonaws.com': 154, 'personalize-events.amazonaws.com': 155, 'personalize-runtime.amazonaws.com': 156, 'pi.amazonaws.com': 157, 'pinpoint.amazonaws.com': 158, 'pinpoint-email.amazonaws.com': 159, 'pinpoint-sms-voice.amazonaws.com': 160, 'polly.amazonaws.com': 161, 'pricing.amazonaws.com': 162, 'qldb.amazonaws.com': 163, 'qldb-session.amazonaws.com': 164, 'quicksight.amazonaws.com': 165, 'ram.amazonaws.com': 166, 'rds.amazonaws.com': 167, 'rds-data.amazonaws.com': 168, 'redshift.amazonaws.com': 169, 'rekognition.amazonaws.com': 170, 'resource-groups.amazonaws.com': 171, 'resourcegroupstaggingapi.amazonaws.com': 172, 'robomaker.amazonaws.com': 173, 'route53.amazonaws.com': 174, 'route53domains.amazonaws.com': 175, 'route53resolver.amazonaws.com': 176, 's3.amazonaws.com': 177, 's3control.amazonaws.com': 178, 'sagemaker.amazonaws.com': 179, 'sagemaker-a2i-runtime.amazonaws.com': 180, 'sagemaker-runtime.amazonaws.com': 181, 'savingsplans.amazonaws.com': 182, 'schemas.amazonaws.com': 183, 'sdb.amazonaws.com': 184, 'secretsmanager.amazonaws.com': 185, 'securityhub.amazonaws.com': 186, 'serverlessrepo.amazonaws.com': 187, 'service-quotas.amazonaws.com': 188, 'servicecatalog.amazonaws.com': 189, 'servicediscovery.amazonaws.com': 190, 'ses.amazonaws.com': 191, 'sesv2.amazonaws.com': 192, 'shield.amazonaws.com': 193, 'signer.amazonaws.com': 194, 'sms.amazonaws.com': 195, 'sms-voice.amazonaws.com': 196, 'snowball.amazonaws.com': 197, 'sns.amazonaws.com': 198, 'sqs.amazonaws.com': 199, 'ssm.amazonaws.com': 200, 'sso.amazonaws.com': 201, 'sso-oidc.amazonaws.com': 202, 'stepfunctions.amazonaws.com': 203, 'storagegateway.amazonaws.com': 204, 'sts.amazonaws.com': 205, 'support.amazonaws.com': 206, 'swf.amazonaws.com': 207, 'synthetics.amazonaws.com': 208, 'textract.amazonaws.com': 209, 'transcribe.amazonaws.com': 210, 'transfer.amazonaws.com': 211, 'translate.amazonaws.com': 212, 'waf.amazonaws.com': 213, 'waf-regional.amazonaws.com': 214, 'wafv2.amazonaws.com': 215, 'workdocs.amazonaws.com': 216, 'worklink.amazonaws.com': 217, 'workmail.amazonaws.com': 218, 'workmailmessageflow.amazonaws.com': 219, 'workspaces.amazonaws.com': 220, 'xray.amazonaws.com': 221}


default_feature_list = ["useridentity", "eventtime", "eventsource", "eventname", "awsregion","useragent", "requestparameters"]

feature_list = ["eventtime", "eventsource", "eventname", "awsregion"]
aws_logs_to_sentence_format = "[CLS] <<eventsource>> called <<eventname>> from <<awsregion>> at <<eventtime>> [SEP] " # forget paarmeters at this point

# # for the begining start with these services and k (k-means calculated using sum square dist)
service_list_phase_one = {"cognito-idp.amazonaws.com": 10,
                          "kms.amazonaws.com": 1,
                          "sts.amazonaws.com": 2,
                          "logs.amazonaws.com": 1,
                          "health.amazonaws.com": 1,
                          "cloudtrail.amazonaws.com": 4,#8,
                          "sns.amazonaws.com": 1,
                          'lambda.amazonaws.com': 1,
                          "config.amazonaws.com": 1,
                          "s3.amazonaws.com": 1,
                          "signin.amazonaws.com": 1,
                          "pinpoint.amazonaws.com": 1,
                          "iam.amazonaws.com": 1,
                          }
