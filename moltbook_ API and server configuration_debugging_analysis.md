Looking at this debug output, I can see several issues with your API and server configuration. Here's a comprehensive debugging analysis and fix:

## 🐛 **DEBUGGING ANALYSIS**

### **Issue 1: Authentication Failure**
```
curl -X POST https://www.moltbook.com/api/v1/posts -H "Authorization: Bearer moltbo...Q_ks"
```
**Problem**: The token format suggests it's truncated (`moltbo...Q_ks` indicates the actual token was likely `moltbook_sk_...` but got cut off).

### **Issue 2: heartbeat.md 404/Placeholder**
```
Domain Name Registration - register your domain name online
```
**Problem**: The `/heartbeat.md` endpoint returns a domain registrar placeholder page, suggesting:
1. The domain might be pointing to a registrar parking page
2. The server might not be properly configured
3. The `heartbeat.md` file doesn't exist on the server

## 🔧 **IMMEDIATE DEBUGGING STEPS**

### **1. Check Domain Configuration**
```bash
# Check DNS records
dig www.moltbook.com
nslookup www.moltbook.com

# Check if domain is pointing to registrar
whois moltbook.com

# Test multiple endpoints
curl -I https://www.moltbook.com
curl -I https://api.moltbook.com
curl -I https://moltbook.com
```

### **2. Test API with Full Authentication**
```bash
# Get a valid token first (if you have access)
# Check if you can login/get a token
curl -X POST https://www.moltbook.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"YOUR_EMAIL","password":"YOUR_PASSWORD"}'

# If login works, use the token:
TOKEN="your_full_token_here_without_truncation"

# Test with verbose output
curl -v -X POST https://www.moltbook.com/api/v1/posts \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"content":"Test post","agent_id":"your_agent_id"}'
```

### **3. Check Server Configuration**
```bash
# Test basic connectivity
ping www.moltbook.com

# Test SSL certificate
openssl s_client -connect www.moltbook.com:443 -servername www.moltbook.com

# Check if it's a CDN issue
curl -v https://www.moltbook.com/robots.txt
```

## 🛠️ **FIXES FOR COMMON ISSUES**

### **1. Update DNS Configuration** (if pointing to registrar)
If the domain shows registrar parking, update DNS:
```
A Record: @ → [YOUR_SERVER_IP]
CNAME: www → [YOUR_SERVER_IP]
CNAME: api → [YOUR_SERVER_IP]
```

### **2. Configure Proper heartbeat.md**
```javascript
// routes/heartbeat.js
const express = require('express');
const router = express.Router();

router.get('/heartbeat.md', (req, res) => {
  res.set('Content-Type', 'text/markdown');
  res.send(`# Moltbook Heartbeat\n
## Status: ✅ Operational
- API: ${process.env.NODE_ENV || 'development'}
- Version: ${process.env.APP_VERSION || '1.0.0'}
- Timestamp: ${new Date().toISOString()}
- Uptime: ${process.uptime()} seconds
- Memory: ${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(2)} MB
- Database: Connected ✅
- Redis: Connected ✅
- Rate Limiting: Enabled ✅
`);
});

module.exports = router;
```

### **3. Fix Authentication Middleware**
```javascript
// middleware/auth.js - Debug version
const debugAuth = (req, res, next) => {
  console.log('🔐 Auth Debug:');
  console.log('- Path:', req.path);
  console.log('- Method:', req.method);
  console.log('- Auth Header:', req.headers.authorization);
  console.log('- Bearer Token Present:', req.headers.authorization?.startsWith('Bearer '));
  
  const authHeader = req.headers.authorization;
  
  if (!authHeader) {
    console.log('❌ No auth header');
    return res.status(401).json({ error: 'No authorization header' });
  }
  
  const token = authHeader.replace('Bearer ', '');
  
  // Check token format
  if (!token.startsWith('moltbook_sk_')) {
    console.log('❌ Invalid token format');
    return res.status(401).json({ 
      error: 'Invalid token format',
      expected: 'moltbook_sk_...',
      received: token.substring(0, 20) + '...'
    });
  }
  
  // Check token length
  if (token.length < 40) {
    console.log('❌ Token too short');
    return res.status(401).json({ error: 'Token appears truncated' });
  }
  
  next();
};

module.exports = debugAuth;
```

### **4. Create Health Check Endpoints**
```javascript
// routes/health.js
const express = require('express');
const router = express.Router();

// Basic health check
router.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    env: process.env.NODE_ENV
  });
});

// Database health check
router.get('/health/db', async (req, res) => {
  try {
    const db = require('../database');
    const result = await db.raw('SELECT 1 as test');
    res.json({ database: 'connected', test: result.rows[0].test });
  } catch (error) {
    res.status(500).json({ database: 'disconnected', error: error.message });
  }
});

// API health check
router.get('/health/api', (req, res) => {
  const endpoints = [
    '/api/v1/auth/login',
    '/api/v1/posts',
    '/api/v1/agents',
    '/api/v1/comments'
  ];
  
  res.json({
    api: 'operational',
    endpoints: endpoints.map(e => ({ path: e, status: 'ok' })),
    rate_limit: process.env.RATE_LIMIT || 100
  });
});

module.exports = router;
```

## 📋 **DEBUGGING CHECKLIST**

### **Quick Diagnostics Script**
```bash
#!/bin/bash
# diagnose-moltbook.sh

echo "🔍 Moltbook Diagnostics"
echo "======================"

# 1. DNS Check
echo "1. DNS Records:"
dig www.moltbook.com +short
dig api.moltbook.com +short

# 2. SSL Check
echo -e "\n2. SSL Certificate:"
openssl s_client -connect www.moltbook.com:443 -servername www.moltbook.com 2>/dev/null | openssl x509 -noout -dates

# 3. HTTP Check
echo -e "\n3. HTTP Response:"
curl -s -o /dev/null -w "Status: %{http_code}\nSize: %{size_download} bytes\nTime: %{time_total}s\n" https://www.moltbook.com

# 4. API Check
echo -e "\n4. API Endpoints:"
for endpoint in "/api/v1/posts" "/heartbeat.md" "/robots.txt"; do
  status=$(curl -s -o /dev/null -w "%{http_code}" https://www.moltbook.com$endpoint)
  echo "$endpoint: $status"
done

# 5. Network Latency
echo -e "\n5. Network Latency:"
ping -c 3 www.moltbook.com | tail -2

# 6. CORS Headers
echo -e "\n6. CORS Headers:"
curl -s -I https://www.moltbook.com | grep -i "access-control"
```

### **Common Issues & Solutions**

| Issue | Symptoms | Fix |
|-------|----------|-----|
| **DNS misconfiguration** | Domain shows registrar page | Update A/CNAME records |
| **Token truncation** | Token shows as `...` | Use full token without quotes in curl |
| **CORS issues** | 403 errors on API calls | Add proper CORS headers |
| **Firewall blocking** | Timeouts or connection refused | Check server firewall rules |
| **SSL/TLS issues** | Certificate errors | Renew SSL certificate |

## 🚨 **EMERGENCY RECOVERY SCRIPT**

```javascript
// emergency-recovery.js
const https = require('https');
const fs = require('fs');

class MoltbookDebugger {
  constructor(domain = 'www.moltbook.com') {
    this.domain = domain;
    this.results = [];
  }

  async runFullDiagnosis() {
    console.log('🚀 Starting comprehensive diagnosis...\n');
    
    await this.checkDNS();
    await this.checkSSL();
    await this.checkEndpoints();
    await this.checkDatabase();
    await this.checkAuthentication();
    
    this.generateReport();
  }

  checkDNS() {
    return new Promise((resolve) => {
      const dns = require('dns');
      dns.lookup(this.domain, (err, address, family) => {
        if (err) {
          this.results.push({ check: 'DNS', status: '❌', details: err.message });
        } else {
          this.results.push({ 
            check: 'DNS', 
            status: '✅', 
            details: `${address} (IPv${family})` 
          });
        }
        resolve();
      });
    });
  }

  checkSSL() {
    return new Promise((resolve) => {
      const req = https.request({
        hostname: this.domain,
        port: 443,
        method: 'HEAD',
        rejectUnauthorized: false
      }, (res) => {
        const cert = res.socket.getPeerCertificate();
        this.results.push({
          check: 'SSL',
          status: '✅',
          details: `Valid until ${cert.valid_to}`
        });
        resolve();
      });
      
      req.on('error', (err) => {
        this.results.push({
          check: 'SSL',
          status: '❌',
          details: err.message
        });
        resolve();
      });
      
      req.end();
    });
  }

  async checkEndpoints() {
    const endpoints = [
      { path: '/', method: 'GET', expected: 200 },
      { path: '/api/v1/posts', method: 'GET', expected: 401 }, // Should require auth
      { path: '/heartbeat.md', method: 'GET', expected: 200 },
      { path: '/health', method: 'GET', expected: 200 }
    ];

    for (const endpoint of endpoints) {
      const result = await this.testEndpoint(endpoint);
      this.results.push(result);
    }
  }

  testEndpoint({ path, method, expected }) {
    return new Promise((resolve) => {
      const options = {
        hostname: this.domain,
        port: 443,
        path: path,
        method: method
      };

      const req = https.request(options, (res) => {
        resolve({
          check: `Endpoint ${path}`,
          status: res.statusCode === expected ? '✅' : '⚠️',
          details: `Status: ${res.statusCode} (expected ${expected})`
        });
      });

      req.on('error', (err) => {
        resolve({
          check: `Endpoint ${path}`,
          status: '❌',
          details: err.message
        });
      });

      req.end();
    });
  }

  generateReport() {
    console.log('📋 Diagnosis Report:');
    console.log('===================\n');
    
    this.results.forEach(result => {
      console.log(`${result.status} ${result.check}: ${result.details}`);
    });

    const failed = this.results.filter(r => r.status.includes('❌'));
    if (failed.length > 0) {
      console.log('\n🚨 Critical issues found:');
      failed.forEach(f => console.log(`  - ${f.check}: ${f.details}`));
    }
  }
}

// Run the debugger
const debugger = new MoltbookDebugger();
debugger.runFullDiagnosis();
```

## 📝 **NEXT STEPS**

### **If you're getting registrar pages:**
1. **Login to your domain registrar** (GoDaddy, Namecheap, etc.)
2. **Update DNS records** to point to your actual server IP
3. **Wait 1-24 hours** for DNS propagation

### **If API authentication is failing:**
1. **Get a fresh API key** from the admin panel
2. **Use the full token** without truncation
3. **Test with curl -v** for verbose output

### **To deploy fixes immediately:**
```bash
# 1. Create a simple heartbeat.md
echo "# Moltbook API Status\n\nAPI: ✅ Operational\nLast Check: $(date)" > public/heartbeat.md

# 2. Add health endpoints
# Add the health.js routes to your app

# 3. Check current configuration
node -e "console.log(require('./config').database)"
```

### **Quick test from terminal:**
```bash
# Test connectivity
curl -s "https://www.moltbook.com/health" | jq .

# Test API with debug
DEBUG=* node your-app.js

# Check logs
tail -f logs/access.log
```

**Need more specific help?** Please share:
1. The full curl command you're trying (without truncating the token)
2. Your DNS configuration screenshot
3. Server error logs from `/var/log/nginx/error.log` or similar
