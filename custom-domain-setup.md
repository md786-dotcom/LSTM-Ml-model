# Custom Domain Setup Guide

## For Root Domain (aistockpredict.com)

### In Render:
1. Add custom domain: `aistockpredict.com`
2. Render provides A records like:
   - `216.24.57.1`
   - `216.24.57.2`

### In Namecheap:
1. Go to Domain List → Manage → Advanced DNS
2. Delete existing A records for @ 
3. Add new A records:
   ```
   Type: A Record
   Host: @
   Value: 216.24.57.1
   TTL: Automatic
   
   Type: A Record  
   Host: @
   Value: 216.24.57.2
   TTL: Automatic
   ```

## For Subdomain (app.aistockpredict.com)

### In Render:
1. Add custom domain: `app.aistockpredict.com`

### In Namecheap:
1. Go to Advanced DNS
2. Add CNAME record:
   ```
   Type: CNAME Record
   Host: app
   Value: stock-prediction-bot.onrender.com.
   TTL: Automatic
   ```

## For Both www and non-www

Add both in Render:
- `aistockpredict.com`
- `www.aistockpredict.com`

In Namecheap, add:
1. A records for @ (root domain)
2. CNAME for www:
   ```
   Type: CNAME
   Host: www
   Value: stock-prediction-bot.onrender.com.
   TTL: Automatic
   ```

## Verification Steps

1. Wait 5-30 minutes for DNS propagation
2. Check status in Render dashboard (will show "Verified")
3. Test your domain:
   - https://aistockpredict.com
   - https://www.aistockpredict.com

## Troubleshooting

- **Not working after 1 hour?** 
  - Check if you added the trailing dot in CNAME
  - Verify you removed old DNS records
  - Use DNS checker: https://dnschecker.org

- **SSL Certificate Error?**
  - Render auto-provisions SSL, just wait a bit
  - Make sure domain is verified in Render

- **Namecheap DNS not updating?**
  - Make sure you're using Namecheap BasicDNS (not custom nameservers)
  - Clear browser cache and try incognito mode