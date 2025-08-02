import { test, expect } from '@playwright/test'

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Mock API responses
    await page.route('**/api/v1/auth/login', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          access_token: 'mock-token',
          token_type: 'bearer',
        }),
      })
    })

    await page.route('**/api/v1/experiments*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          items: [
            {
              experiment_id: 'exp_20240101_120000',
              start_time: '2024-01-01T12:00:00',
              end_time: '2024-01-01T12:30:00',
              total_rounds: 10,
              total_games: 450,
              total_api_calls: 100,
              total_cost: 5.50,
              status: 'completed',
            },
            {
              experiment_id: 'exp_20240101_130000',
              start_time: '2024-01-01T13:00:00',
              end_time: '2024-01-01T13:45:00',
              total_rounds: 15,
              total_games: 675,
              total_api_calls: 150,
              total_cost: 8.25,
              status: 'completed',
            },
          ],
          total: 2,
          page: 1,
          page_size: 20,
          total_pages: 1,
        }),
      })
    })
  })

  test('should display experiment dashboard', async ({ page }) => {
    await page.goto('/')
    
    // Check page title
    await expect(page.locator('h1')).toContainText('Experiment Dashboard')
    
    // Check stats cards are displayed
    await expect(page.locator('text=Total Experiments')).toBeVisible()
    await expect(page.locator('text=Running')).toBeVisible()
    await expect(page.locator('text=Completed')).toBeVisible()
    await expect(page.locator('text=Total Cost')).toBeVisible()
  })

  test('should display experiment list', async ({ page }) => {
    await page.goto('/')
    
    // Wait for experiments to load
    await page.waitForSelector('text=exp_20240101_120000')
    
    // Check experiment cards
    await expect(page.locator('text=exp_20240101_120000')).toBeVisible()
    await expect(page.locator('text=exp_20240101_130000')).toBeVisible()
    
    // Check experiment details
    await expect(page.locator('text=10 rounds').first()).toBeVisible()
    await expect(page.locator('text=450 games').first()).toBeVisible()
    await expect(page.locator('text=$5.50').first()).toBeVisible()
  })

  test('should filter experiments by search', async ({ page }) => {
    await page.goto('/')
    
    // Wait for experiments to load
    await page.waitForSelector('text=exp_20240101_120000')
    
    // Search for specific experiment
    await page.fill('input[placeholder="Search experiments..."]', '120000')
    
    // Check filtered results
    await expect(page.locator('text=exp_20240101_120000')).toBeVisible()
    await expect(page.locator('text=exp_20240101_130000')).not.toBeVisible()
  })

  test('should sort experiments', async ({ page }) => {
    await page.goto('/')
    
    // Change sort field
    await page.selectOption('select', 'total_cost')
    
    // Check that API was called with sort parameters
    await page.waitForRequest(request => 
      request.url().includes('sort_by=total_cost')
    )
  })

  test('should toggle sort order', async ({ page }) => {
    await page.goto('/')
    
    // Click sort order button
    await page.click('button[aria-label*="Sort"]')
    
    // Check that API was called with different sort order
    await page.waitForRequest(request => 
      request.url().includes('sort_order=asc')
    )
  })

  test('should navigate to experiment details', async ({ page }) => {
    await page.goto('/')
    
    // Wait for experiments to load
    await page.waitForSelector('text=exp_20240101_120000')
    
    // Click view details
    await page.click('text=View Details')
    
    // Check navigation
    await expect(page).toHaveURL(/\/experiments\/exp_20240101_120000/)
  })

  test('should handle empty state', async ({ page }) => {
    // Override route to return empty results
    await page.route('**/api/v1/experiments*', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          items: [],
          total: 0,
          page: 1,
          page_size: 20,
          total_pages: 0,
        }),
      })
    })

    await page.goto('/')
    
    // Check empty state message
    await expect(page.locator('text=No experiments found')).toBeVisible()
    await expect(page.locator('text=Start your first experiment')).toBeVisible()
  })

  test('should handle API errors gracefully', async ({ page }) => {
    // Override route to return error
    await page.route('**/api/v1/experiments*', async route => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: 'Internal server error',
        }),
      })
    })

    await page.goto('/')
    
    // Should show error toast or message
    await expect(page.locator('text=Failed to load experiments')).toBeVisible()
  })

  test('should display loading state', async ({ page }) => {
    let resolveResponse: () => void
    const responsePromise = new Promise<void>(resolve => {
      resolveResponse = resolve
    })

    // Delay API response
    await page.route('**/api/v1/experiments*', async route => {
      await responsePromise
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          items: [],
          total: 0,
          page: 1,
          page_size: 20,
          total_pages: 0,
        }),
      })
    })

    await page.goto('/')
    
    // Check loading skeletons are visible
    await expect(page.locator('.animate-pulse').first()).toBeVisible()
    
    // Resolve the response
    resolveResponse!()
    
    // Wait for loading to complete
    await page.waitForSelector('.animate-pulse', { state: 'hidden' })
  })
})