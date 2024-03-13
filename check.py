import boto3

iam = boto3.client('iam')
response = iam.list_roles()

roles = response['Roles']
for role in roles:
    if 'rec-team-role' in role['Arn']:
        print(role['RoleName'])
