import { retry } from '@/utils/retry'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

interface ApiConfig {
  baseUrl: string
  token?: string | null
}

class ApiClient {
  private config: ApiConfig
  
  constructor(config: ApiConfig) {
    this.config = config
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    }
    
    const token = this.config.token || localStorage.getItem('auth_token')
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }
    
    return headers
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.config.baseUrl}${endpoint}`
    const response = await fetch(url, {
      ...options,
      headers: {
        ...this.getHeaders(),
        ...options.headers,
      },
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({}))
      throw new Error(error.detail || `Request failed: ${response.statusText}`)
    }

    return response.json()
  }

  async login(username: string, password: string) {
    const formData = new FormData()
    formData.append('username', username)
    formData.append('password', password)

    const response = await fetch(`${this.config.baseUrl}/auth/login`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error('Invalid credentials')
    }

    const data = await response.json()
    localStorage.setItem('auth_token', data.access_token)
    return data
  }

  async logout() {
    try {
      await this.request('/auth/logout', { method: 'POST' })
    } finally {
      localStorage.removeItem('auth_token')
    }
  }

  async getExperiments(params: {
    page?: number
    pageSize?: number
    sortBy?: string
    sortOrder?: 'asc' | 'desc'
  } = {}) {
    const queryParams = new URLSearchParams({
      page: String(params.page || 1),
      page_size: String(params.pageSize || 20),
      sort_by: params.sortBy || 'start_time',
      sort_order: params.sortOrder || 'desc',
    })

    return retry(() => 
      this.request<{
        items: any[]
        total: number
        page: number
        page_size: number
        total_pages: number
      }>(`/experiments?${queryParams}`)
    )
  }

  async getExperiment(experimentId: string) {
    return retry(() => 
      this.request<any>(`/experiments/${experimentId}`)
    )
  }

  async getRoundData(experimentId: string, roundNum: number) {
    return retry(() => 
      this.request<any>(`/experiments/${experimentId}/rounds/${roundNum}`)
    )
  }

  async getCurrentUser() {
    return this.request<any>('/auth/me')
  }
}

export const api = new ApiClient({
  baseUrl: API_BASE_URL,
})