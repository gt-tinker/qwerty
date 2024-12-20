# Before you can use this, you'll need to do a few things:
# 1. Boot into Windows. My condolences
# 2. Make sure Docker Desktop is set to use Windows containers, not Linux
#    containers. (You can right-click Docker Desktop in the notification area
#    to swap it over if if needed.)
# 3. Authenticate with the AWS CLI via `aws configure'

$aws_account_id = '917270012582'
$aws_region = 'us-east-1'
$aws_ecr_repo_name = 'qwerty'
$image_name = 'qwerty-llvm-vs'
$image_sha = docker image list -q $image_name

if ($image_sha) {
    echo "=====> Image $image_name already exists, $image_sha. Skipping rebuild"
} else {
    echo "=====> Image $image_name does not exist, building..."
    docker build -t qwerty-llvm-vs .
    $image_sha = docker image list -q $image_name
}

echo "=====> Tagging and pushing $image_name to AWS ECR..."

$aws_hostname = "$aws_account_id.dkr.ecr.$aws_region.amazonaws.com"

aws ecr get-login-password --region $aws_region | docker login --username AWS --password-stdin $aws_hostname

$full_image_name = "$aws_hostname/${aws_ecr_repo_name}:$image_name"

docker tag $image_sha $full_image_name
docker push $full_image_name
